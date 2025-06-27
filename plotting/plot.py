import pandas as pd
import altair as alt
from gradio.components.plot import PlotData
from typing import Optional, Union, List
from gradio.events import Events



def CustomLinePlot(
    value: pd.DataFrame,
    x_col: str = "Timestamp",
    x_label: str = "Date",
    y_col: str = "Score",
    y_label: str = "Sentiment Score",
    color: str = "SentScore Label",
    title: str = "Sentiment Score Time Series",
    padding: Optional[dict] = None,
    title_font_size: int = 24,
    axis_title_font_size: int = 16,
    axis_label_font_size: int = 12
) -> PlotData:
    """
    Takes a melted DataFrame with columns [x_col, color, y_col] and returns
    an Altair PlotData with hoverable points, zoom/pan, and adjustable font sizes.
    """
    if value is None or value.empty or not all(col in value.columns for col in [x_col, y_col, color]):
        empty_chart = alt.Chart(pd.DataFrame({x_col: [], color: [], y_col: []})).mark_line()
        return PlotData(type="altair", plot=empty_chart.to_json())

    # Build the base encoding
    base = alt.Chart(value).encode(
        x=alt.X(
            f"{x_col}:T" if pd.api.types.is_datetime64_any_dtype(value[x_col]) else x_col,
            title=x_label,
            axis=alt.Axis(
                labelAngle=45,
                titleFontSize=axis_title_font_size,
                labelFontSize=axis_label_font_size
            )
        ),
        y=alt.Y(
            f"{y_col}:Q",
            title=y_label,
            axis=alt.Axis(
                titleFontSize=axis_title_font_size,
                labelFontSize=axis_label_font_size
            )
        ),
        color=alt.Color(
            field=color,
            type="nominal",
            title="",
            legend=alt.Legend(orient="top", labelFontSize=axis_label_font_size)
        ),
        tooltip=[
            alt.Tooltip(
                f"{x_col}:T" if pd.api.types.is_datetime64_any_dtype(value[x_col]) else x_col,
                title=x_label
            ),
            alt.Tooltip(f"{color}:N", title="Series"),
            alt.Tooltip(f"{y_col}:Q", title=y_label)
        ]
    )

    # Layer: line + invisible‐until‐hover points
    lines = base.mark_line(clip=True).encode(size=alt.value(2))
    points = base.mark_point(clip=True).encode(
        opacity=alt.value(0),
        size=alt.value(30)
    )

    highlight = alt.selection_point(
        on="mouseover", fields=[color], nearest=True, clear="mouseout", empty=False
    )
    points = points.add_params(highlight)
    lines = lines.encode(size=alt.condition(highlight, alt.value(4), alt.value(2)))

    properties = {
        "title": title,
        "height": 600,
        "width": 900,
        "padding": {"left": 60, "right": 20, "top": 40, "bottom": 50} if padding is None else padding
    }
    chart = (lines + points).properties(**properties).configure_title(fontSize=title_font_size).interactive()

    # Optional: add an interval brush on X
    selector = alt.selection_interval(
        encodings=["x"],
        mark=alt.BrushConfig(fill="gray", fillOpacity=0.3, stroke="none"),
        name="brush"
    )
    chart = chart.add_params(selector)

    return PlotData(type="altair", plot=chart.to_json())