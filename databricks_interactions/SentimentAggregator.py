import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

from dataclasses import dataclass, field
import torch
from typing import Union, List, Dict, Set, Tuple, FrozenSet, Literal
from sentence_transformers import SentenceTransformer 
import pandas as pd 
import json


# from databricks.connect import DatabricksSession 
 

@dataclass
class SentAggConfig:
  model_load_path: str = "/dbfs/mnt/sedl/TX/News/st_embedding_model_e5_small"
  device: str = "cuda" if torch.cuda.is_available() else "cpu"
  similarity_threshold: float = 0.7 # Added similarity threshold
  _encoder_instance: SentenceTransformer = None

  @property
  def encoder(self):
    if SentAggConfig._encoder_instance is None:
        SentAggConfig._encoder_instance = SentenceTransformer(self.model_load_path, device=self.device)
    return SentAggConfig._encoder_instance



class SentAgg:
  def __init__(self,
               spark: SparkSession = None, #
               config: Union[SentAggConfig, dict] = SentAggConfig(), 
               ) -> None:
    
    self.freq_mapping = dict(zip(["Monthly", "Weekly (M-SUN)", "Daily"], ["M", "W", "D"]))
    
    
    self.spark = spark 

    if config is None:
        self.config = SentAggConfig()
    elif isinstance(config, dict):
      try:
        self.config = SentAggConfig(**config)
      except Exception as e:

        raise ValueError(f"Config dictionary is not sufficient or invalid for SentAggConfig: {e}")
    else:
      self.config = config
    
    if not isinstance(self.config, SentAggConfig):
        raise TypeError("config must be an instance of SentAggConfig or a compatible dict.")


    self.filter_mapping = self._load_filter_mapping()

  def _load_filter_mapping(self):
    with open("input_validation_data/description_to_rcsCode.json", "r") as f:
      topic_mapping =  json.load(f)
    with open("input_validation_data/name_to_permid.json", "r") as jf:
        permid_mapping =  json.load(jf)
    
    topic_mapping.update(permid_mapping)
    return topic_mapping
       

  def _validate_params(self, params: Dict[str, List[Set[str]]]) -> Tuple[Dict[str, List[Set[str]]], Dict[str, List[Tuple[str, FrozenSet[str]]]]]:
    """
    Validates parameters, separating query-based conditions from topic-based conditions.
    
    Args:
        params: Dict where keys are labels and values are lists of sets.
                Each set contains joint conditions (strings).
                Conditions starting with "query: " are treated as semantic search queries.

    Returns:
        A tuple containing:
        - non_query_conditions: Dict similar to params, but with query strings removed.
                                Labels/sets that become empty are pruned.
        - query_details: Dict where keys are query strings (e.g., "query: text").
                         Values are lists of tuples: (label, frozenset_of_other_conditions_in_same_set).
                         This stores the context of where each query appeared.
    """
    non_query_conditions = {}
    query_details = {}

    for label, list_of_condition_sets in params.items():
        processed_list_of_sets_for_label = []
        for original_set in list_of_condition_sets:
            current_set_non_query_items = set()
            current_set_query_items_with_context = []

            for item in original_set:
                if isinstance(item, str) and item.startswith("query: "):
                    other_items_in_set = frozenset(i for i in original_set if i != item and not (isinstance(i, str) and i.startswith("query: ")))
                    current_set_query_items_with_context.append({'query_string': item, 'context': (label, other_items_in_set)})
                elif isinstance(item, str): 
                  filter_item = self.filter_mapping.get(item,  None)
                  if filter_item:
                    if isinstance(filter_item, list):
                      for permid in filter_item: 
                        current_set_non_query_items.add(permid)
                    if isinstance(filter_item, str):
                      current_set_non_query_items.add(filter_item)
                      

            if current_set_non_query_items:
                processed_list_of_sets_for_label.append(current_set_non_query_items)
            
            for query_data in current_set_query_items_with_context:
                query_str = query_data['query_string']
                context = query_data['context']
                if query_str not in query_details:
                    query_details[query_str] = []
                query_details[query_str].append(context)

        if processed_list_of_sets_for_label: 
            non_query_conditions[label] = processed_list_of_sets_for_label
            
    return non_query_conditions, query_details

  
  def _embed_queries(self, queries: List[str]) -> Dict[str, List[float]]:
    """Encodes a list of query strings into embeddings."""
    if not queries:
      return {}
      
   
    embeddings_tensor = self.config.encoder.encode(
                      queries,
                      convert_to_tensor=True,
                      normalize_embeddings=True, 
                      #show_progress_bar=True # Optional
                  )
    embeddings_list = embeddings_tensor.cpu().tolist()
    return {query: emb for query, emb in zip(queries, embeddings_list)}


  def query_news(self, 
                 params: Dict[str, List[Set[str]]], 
                 freq = Literal["Monthly", "Weekly (M-SUN)", "Daily"], 
                 spark : "SparkSession" = None):
    """
    Queries news data based on subject topics and semantic query embeddings.
    """
    yield "Validate Params"
    freq = self.freq_mapping[freq]
    non_query_conditions, query_details = self._validate_params(params)

    unique_query_strings = list(query_details.keys())
    yield "Embeddings queries if there are any"
    query_embeddings_map = self._embed_queries(unique_query_strings)
    ordered_query_embedding_vectors = [query_embeddings_map[q_str] for q_str in unique_query_strings]


    @F.udf(T.ArrayType(T.FloatType()))
    def cs_udf(news_embedding_col: List[float]):
      if not news_embedding_col or not ordered_query_embedding_vectors:
          return [] 
      news_vec = torch.tensor(news_embedding_col, dtype=torch.float16, device='cpu').unsqueeze(0) # (1, D)
      
      queries_vec = torch.tensor(ordered_query_embedding_vectors, dtype=torch.float16, device='cpu') # (N_queries, D)

      if queries_vec.numel() == 0: # No queries
          return []

      cos_sim = torch.nn.functional.cosine_similarity(news_vec, queries_vec, dim=1) # Output shape (N_queries)
      return cos_sim.tolist()

    current_similarity_threshold = self.config.similarity_threshold #TODO: call a reasoning model here on a subsample of the thresholds

    @F.udf(T.ArrayType(T.StringType()))
    def screen_news_udf(subjects_col: List[str], cos_sim_scores_col: List[float]):
      if subjects_col is None: # Handle null subjects_col if necessary
          subjects_col = []
      
      news_subjects_set = set(subjects_col)
      satisfied_labels = set()

      for label, list_of_condition_sets in non_query_conditions.items():
        for condition_set in list_of_condition_sets: 
          if condition_set  .issubset(news_subjects_set):
            satisfied_labels.add(label)

      for i, query_str in enumerate(unique_query_strings):
        if i < len(cos_sim_scores_col): # Ensure index is valid
            similarity_score = cos_sim_scores_col[i]
            if similarity_score >= current_similarity_threshold:
                contexts = query_details.get(query_str, [])
                for label, other_conditions_in_set in contexts:
                    if not other_conditions_in_set or other_conditions_in_set.issubset(news_subjects_set):
                        satisfied_labels.add(label)
      
      return sorted(list(satisfied_labels)) 


    # TODO: Consider making the path a parameter or part of the config
    try: 
      #logger.info("spark successfully connected")"/mnt/sedl/TX/News/TRNA_LIVE_LINKED_2024_EMBEDDED.delta"
      df = spark.read.format("delta").load("/mnt/sedl/TX/News/TRNA_LIVE_LINKED.delta") \
        .select(
          "timestamp",
          "subjects",
          "sentimentClass",
          "sentimentPositive",
          "sentimentNegative",
          # "sentimentNeutral", 
          #"embedding",
          # "RefFound", "HLOnly" 
        ) \
        #.filter(F.col("timestamp") >= "2024-01-01")
      yield "Loaded Spark Dataframe"
      
      if unique_query_strings:
          yield "Computing query - news similarity"
          df = df.withColumn("cos_sim_scores", cs_udf(F.col("embedding")))
          yield "Compute query - news similarity complete"
      else: 
          df = df.withColumn("cos_sim_scores", F.lit([]).cast(T.ArrayType(T.FloatType())))
          yield "Skipping query embedding step since no queries were encountered"

      df = df.withColumn("selected_labels", screen_news_udf(F.col("subjects"), F.col("cos_sim_scores")))
      df = df.filter(F.size(F.col("selected_labels")) > 0)

      yield "Filtered News to contain relevant labelled news"
      df = df.withColumn("Timestamp", F.col("timestamp").cast("date")).cache()
      """
      df = df.select("Timestamp",
                  "selected_labels", 
                  "body",
                  "sentimentClass",
                  "sentimentPositive",
                  "sentimentNegative")
                  """
      df_exploded = df.withColumn("selected_label", F.explode("selected_labels"))
      
      agg_df = df_exploded.groupBy("Timestamp", "selected_label").agg(
        F.coalesce( 
          F.sum(
            F.when(F.col("sentimentClass") == 1, F.col("sentimentPositive"))
            .when(F.col("sentimentClass") == -1, -F.col("sentimentNegative"))
            .otherwise(0.0)
          ),
          F.lit(0.0) 
        ).alias("total_sentiment_value"),
        F.count(F.col("sentimentClass")).alias("news_count")
      )

      yield "Convert Pyspark DF to Pandas DF"


      pandas_agg_df = agg_df \
        .select("Timestamp", "selected_label", "total_sentiment_value", "news_count") \
        .toPandas()

      yield "Aggregated Labelled News"
      
      # try:
      #   spark.stop()
      # except Exception as e:
      #   pass
      

      if not pandas_agg_df.empty:
        pandas_agg_df['Timestamp'] = pd.to_datetime(pandas_agg_df['Timestamp'])
        pandas_agg_df.set_index('Timestamp', inplace = True)
        
        
        # Post-aggregation in pandas since most granular pyspark is daily this fits into memory
        pandas_agg_df = pandas_agg_df.groupby([pd.Grouper(freq=freq), 'selected_label'], 
                                              as_index = True).agg(
          SentScoreSum = ('total_sentiment_value', 'sum'),
          news_count = ('news_count', 'sum'))
          
        pandas_agg_df["selected_label"] = pandas_agg_df.index.get_level_values("selected_label")
        pandas_agg_df.index = pandas_agg_df.index.get_level_values("Timestamp")

      

        # Builting pandas div. if news_count is 0 will be pd.NA
        pandas_agg_df["SentScore"] = pandas_agg_df["SentScoreSum"].div(pandas_agg_df["news_count"])
        
        # Pivot dataframes such that their labels become columns
        pandas_count_piv = pd.pivot_table(pandas_agg_df, index = "Timestamp", values= "news_count", columns = "selected_label").sort_index()
        pandas_sent_piv = pd.pivot_table(pandas_agg_df, index = "Timestamp", values= "SentScore", columns = "selected_label").sort_index()
        
        # Format the dfs such that gradio displays the timestmap index 
        pandas_sent_piv = pandas_sent_piv.rename_axis('Timestamp').reset_index()
        pandas_count_piv = pandas_count_piv.rename_axis('Timestamp').reset_index()
        # logger.info(f"sucessfully retrieved data for params: {params}")
        return {
          "Sentiment Scores" : pandas_sent_piv, 
          "News Counts" : pandas_count_piv, 
          "All Agg Data" : pandas_agg_df,
          # "Mentioned Company Names" : mentioned_c_names, 
          # "Mentioned Company Counts" : mentioned_c_counts, 
          # "Average Return per Label" :  avg_return_per_label, 
          }
      
      else: 
        print("pandas emtpy")
        raise ValueError(f"parmas {non_query_conditions} and unique query strings {unique_query_strings} with pandas dataframe {pandas_agg_df}")
        
    except Exception as e:
      # try:
      #   spark.stop()
      # #logger.info("[SentAgg] spark stopped")
      # except Exception as e:
      #   pass
      return e
    


