"""Dash Dashboard for CNN Text Classification

A modern, professional interface for testing CNN-based text classification.
Maintains prediction functionality while adding sidebars, example buttons,
dual charts, and confidence indicators.
    # Create a placeholder probability card (shown before first prediction)

Run with:
    python app.py

Architecture:
1. Configuration & constants
2. Helper functions (prediction, charts, UI builders)
3. App initialization
4. Styled HTML template
5. Layout (sidebar + main content)
6. Callbacks (user interactions)
"""

from __future__ import annotations

import inspect
import os
from typing import Dict, List, Tuple

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from Predict import predict


# =============================
# Configuration

MODEL_PATH = os.path.join("models", "ablation2_use_adam_optimizer_best.pth")
CONFIG_NAME = "ablation2"
DEFAULT_CLASS_ORDER = ["world", "Sports", "Business", "Sci/Tech"]

# Model information displayed in sidebar
MODEL_INFO = {
    "name": "Character-level CNN",
    "dataset": "AG News Dataset",
    "classes": 4,
    "accuracy": "~89%",
    "input": "Character sequences",
}

# Example texts for each class
    
    # Map confidence level to corresponding color (green/yellow/red)
EXAMPLE_TEXTS = {
    "Sports": "NFL playoff: The Kansas City Chiefs defeated the San Francisco 49ers in a thrilling Super Bowl match.",
    "Business": "Apple Inc. reports record earnings for Q2 2024, driven by strong iPhone sales and services growth.",
    "Sci/Tech": "Artificial intelligence breakthroughs in deep learning allow models to process natural language better.",
    "world": "United Nations calls for climate action as global temperatures continue to rise at unprecedented rates.",
}

# Dark theme with confidence-based colors
THEME = {
    "bg": "#0d1117",
    "sidebar": "#161b22",
    "panel": "#0d1117",
    "panel_light": "#161b22",
    "text": "#e6edf3",
    "muted": "#9aa4b2",
    "accent": "#63a4ff",
    "border": "#384355",
    # Confidence color indicators
    "high_conf": "#49c16d",      
    "med_conf": "#e0ac39",      
    "low_conf": "#ff5d5d",     
}


#  style dictionaries keep the layout readable 
SIDEBAR_ITEM_STYLE = {
    "display": "flex",
    "justifyContent": "space-between",
    "padding": "10px",
    "backgroundColor": THEME["panel"],
    "borderRadius": "6px",
    "marginBottom": "8px",
    "fontSize": "13px",
    "border": f"1px solid {THEME['border']}",
}

SIDEBAR_LABEL_STYLE = {
    "color": THEME["muted"],
    "fontWeight": "500",
}

SIDEBAR_VALUE_STYLE = {
    "color": THEME["accent"],
    "fontWeight": "600",
}

EXAMPLE_BUTTON_STYLE = {
    "display": "block",
    "width": "100%",
    "padding": "10px",
    "marginBottom": "8px",
    "backgroundColor": "#238636",
    "color": "white",
    "border": "none",
    "borderRadius": "6px",
    "fontSize": "12px",
    "fontWeight": "600",
    "cursor": "pointer",
    "transition": "all 200ms ease",
    "textAlign": "left",
    "overflow": "hidden",
    "textOverflow": "ellipsis",
    "whiteSpace": "nowrap",
}

CARD_STYLE = {
    "backgroundColor": THEME["sidebar"],
    "border": f"1px solid {THEME['border']}",
    "borderRadius": "12px",
    "padding": "20px",
    "marginBottom": "20px",
    "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.3)",
    "transition": "all 200ms ease",
}

TEXTAREA_STYLE = {
    "width": "100%",
    "minHeight": "150px",
    "padding": "14px",
    "backgroundColor": THEME["panel"],
    "color": THEME["text"],
    "border": f"1px solid {THEME['border']}",
    "borderRadius": "8px",
    "fontFamily": "'Inter', sans-serif",
    "fontSize": "14px",
    "resize": "vertical",
    "transition": "all 200ms ease",
}

METRIC_BOX_STYLE = {
    "backgroundColor": THEME["panel"],
    "border": f"1px solid {THEME['border']}",
    "borderRadius": "8px",
    "padding": "12px",
    "textAlign": "center",
}

METRIC_LABEL_STYLE = {
    "fontSize": "12px",
    "color": THEME["muted"],
    "marginBottom": "8px",
    "fontWeight": "600",
    "textTransform": "uppercase",
}

METRIC_VALUE_STYLE = {
    "fontSize": "24px",
    "fontWeight": "700",
    "color": THEME["accent"],
}

CONFIDENCE_BADGE_BASE_STYLE = {
    "display": "inline-block",
    "marginTop": "8px",
    "padding": "4px 8px",
    "fontSize": "11px",
    "fontWeight": "700",
    "borderRadius": "4px",
}

CONFIDENCE_BAR_BASE_STYLE = {
    "height": "100%",
    "borderRadius": "999px",
    "transition": "width 500ms ease",
}

TEXT_BLOCK_STYLE = {
    "fontSize": "12px",
    "color": THEME["muted"],
    "lineHeight": "1.6",
    "whiteSpace": "pre-wrap",
}



# ====================================
# Helper: Confidence Color Indicator

def get_confidence_color(confidence: float) -> str:
    """Return color based on confidence level.
    
    Args:
        confidence: float between 0 and 1
        
    Returns:
        Color hex code (green for high, yellow for medium, red for low)
    """
    if confidence >= 0.7:
        return THEME["high_conf"]  # Green
    elif confidence >= 0.4:
        return THEME["med_conf"]   # Yellow/Orange
    else:
        return THEME["low_conf"]   # Red


def get_confidence_label(confidence: float) -> str:
    """Return confidence category label."""
    
    # Return text label for confidence level (High/Medium/Low)
    if confidence >= 0.7:
        return "High"
    elif confidence >= 0.4:
        return "Medium"
    else:
        return "Low"


def build_sidebar_item(label: str, value: str, value_style: dict | None = None) -> html.Div:
    """Create one row in the model information sidebar."""
    return html.Div(
        [
            html.Div(label, className="sidebar-label", style=SIDEBAR_LABEL_STYLE),
            html.Div(value, className="sidebar-value", style=value_style or SIDEBAR_VALUE_STYLE),
        ],
        className="sidebar-item",
        style=SIDEBAR_ITEM_STYLE,
    )


def build_example_button(button_id: str, label: str, text: str) -> html.Button:
    # Create a clickable example text button for the sidebar
    """Create one example button in the sidebar."""
    return html.Button(
        label,
        id=button_id,
        n_clicks=0,
        className="example-btn",
        title=text,
        style=EXAMPLE_BUTTON_STYLE,
    )


def build_metric_box(title: str, content: List[html.Component]) -> html.Div:
    # Create a metric display card (for confidence, meter, etc.)
    """Create a small metric card used in the prediction summary."""
    return html.Div(
        className="metric-box",
        style=METRIC_BOX_STYLE,
        children=[html.Div(title, className="metric-label", style=METRIC_LABEL_STYLE), *content],
    )


def build_confidence_badge_style(color: str) -> Dict[str, str]:
    # Build CSS style dict for confidence badge with specified color
    """Return the badge style for a given confidence color."""
    return {
        **CONFIDENCE_BADGE_BASE_STYLE,
        "backgroundColor": color + "33",
        "color": color,
    }


def build_confidence_bar_style(color: str, width: str) -> Dict[str, str]:
    # Build CSS style dict for confidence progress bar with color and width
    """Return the progress bar style for a given confidence color and width."""
    return {
        **CONFIDENCE_BAR_BASE_STYLE,
        "width": width,
        "background": f"linear-gradient(90deg, {color}, {color}dd)",
    }


def build_empty_probability_card(class_name: str) -> html.Div:
    """Create a placeholder probability card before the first prediction."""
    return html.Div(
        [
            html.Div(class_name, style={"fontWeight": "600", "marginBottom": "6px"}),
            html.Div("--", style={"fontSize": "18px", "fontWeight": "700"}),
        ],
        style={
            "padding": "12px",
            "borderRadius": "8px",
            "border": f"1px solid {THEME['border']}",
            "backgroundColor": THEME["panel_light"],
            "textAlign": "center",
        },
    )


def build_probability_card(class_name: str, probability: float, predicted_label: str) -> html.Div:
    # Create a probability card for one class with highlight if it's the prediction
    """Create one probability card with optional highlight for the prediction."""
    is_predicted = class_name == predicted_label
    conf_color = get_confidence_color(probability)

    return html.Div(
        [
            html.Div(
                class_name,
                style={
                    "fontWeight": "700",
                    "marginBottom": "6px",
                    "color": conf_color if is_predicted else THEME["text"],
                },
            ),
            html.Div(
                f"{probability * 100:.1f}%",
                style={
                    "fontSize": "20px",
                    "fontWeight": "700",
                    "color": conf_color if is_predicted else THEME["text"],
                },
            ),
        ],
        style={
            "padding": "14px",
            "borderRadius": "8px",
            "border": f"2px solid {conf_color if is_predicted else THEME['border']}",
            "backgroundColor": THEME["panel_light"],
            "textAlign": "center",
            "transition": "all 200ms ease",
            "boxShadow": f"0 0 12px {conf_color}66" if is_predicted else "none",
        },
    )


def build_sidebar() -> html.Div:
    # Assemble the left sidebar with model info, example buttons, and instructions
    """Create the left sidebar with model info, examples, and instructions."""
    return html.Div(
        className="sidebar",
        children=[
            html.Div(
                className="sidebar-section",
                children=[
                    html.Div("Model Information", className="sidebar-title"),
                    html.Div(
                        [
                            build_sidebar_item("Name", MODEL_INFO["name"]),
                            build_sidebar_item("Dataset", MODEL_INFO["dataset"]),
                            build_sidebar_item("Classes", str(MODEL_INFO["classes"])),
                            build_sidebar_item("Accuracy", MODEL_INFO["accuracy"]),
                            build_sidebar_item(
                                "Input",
                                MODEL_INFO["input"],
                                value_style={**SIDEBAR_VALUE_STYLE, "fontSize": "11px"},
                            ),
                        ]
                    ),
                ],
            ),
            html.Div(
                className="sidebar-section",
                children=[
                    html.Div("Example Texts", className="sidebar-title"),
                    build_example_button("example-btn-sports", "Sports Example", EXAMPLE_TEXTS["Sports"]),
                    build_example_button("example-btn-business", "Business Example", EXAMPLE_TEXTS["Business"]),
                    build_example_button("example-btn-scitech", "Sci/Tech Example", EXAMPLE_TEXTS["Sci/Tech"]),
                    build_example_button("example-btn-world", "World Example", EXAMPLE_TEXTS["world"]),
                ],
            ),
            html.Div(
                className="sidebar-section",
                children=[
                    html.Div("How to Use", className="sidebar-title"),
                    html.Div(
                        "1. Enter text or click an example\n2. Click Predict button\n3. View results and probabilities",
                        style=TEXT_BLOCK_STYLE,
                    ),
                ],
            ),
        ],
    )


def build_input_card() -> html.Div:
    # Create the text input area and predict button section
    """Create the text input section and predict button."""
    return html.Div(
        className="card",
        style=CARD_STYLE,
        children=[
            html.Div("Input Text", className="card-title"),
            dcc.Textarea(
                id="input-text",
                className="textarea",
                placeholder="Paste or type your text here (up to 1014 characters)...",
                style=TEXTAREA_STYLE,
            ),
            html.Button(" Predict Class", id="predict-btn", n_clicks=0, className="predict-btn"),
            html.Div(id="status-message", style={"marginTop": "12px", "fontSize": "12px", "color": THEME["muted"]}),
        ],
    )


def build_prediction_summary() -> html.Div:
    # Create the prediction result display with confidence metric and badge
    """Create the prediction summary area shown above the charts."""
    return html.Div(
        children=[
            html.Div(
                "Awaiting prediction...",
                id="predicted-class",
                style={
                    "fontSize": "28px",
                    "fontWeight": "700",
                    "marginBottom": "12px",
                    "color": THEME["accent"],
                },
            ),
            html.Div(
                "Click Predict button to classify text.",
                id="prediction-desc",
                style={"fontSize": "13px", "color": THEME["muted"], "marginBottom": "16px"},
            ),
            html.Div(
                className="grid-2",
                children=[
                    build_metric_box(
                        "Confidence",
                        [
                            html.Div("0.00%", id="confidence-text", className="metric-value", style=METRIC_VALUE_STYLE),
                            html.Div("Low", id="confidence-badge", style=build_confidence_badge_style(THEME["low_conf"])),
                        ],
                    ),
                    build_metric_box(
                        "Confidence Meter",
                        [
                            html.Div(
                                className="confidence-meter",
                                style={
                                    "width": "100%",
                                    "height": "10px",
                                    "backgroundColor": THEME["panel"],
                                    "borderRadius": "999px",
                                    "overflow": "hidden",
                                    "border": f"1px solid {THEME['border']}",
                                    "marginTop": "8px",
                                },
                                children=[
                                    html.Div(
                                        id="confidence-bar",
                                        className="confidence-bar",
                                        style=build_confidence_bar_style(THEME["low_conf"], "0%"),
                                    )
                                ],
                            ),
                            html.Div(
                                "0%",
                                id="confidence-label",
                                className="confidence-label",
                                style={"fontSize": "11px", "color": THEME["muted"], "marginTop": "6px", "fontWeight": "500"},
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def build_probability_section() -> html.Div:
    # Create the probability cards grid section
    """Create the class probability card section."""
    return html.Div(
        className="card",
        style=CARD_STYLE,
        children=[
            html.Div("Class Probabilities", className="card-title"),
            html.Div(
                id="probability-cards",
                className="grid-2",
                children=[build_empty_probability_card(class_name) for class_name in DEFAULT_CLASS_ORDER],
            ),
        ],
    )


def build_chart_card(graph_id: str, figure: go.Figure) -> html.Div:
    # Wrap a Plotly graph in a card container
    """Create one chart card used for the bar and pie charts."""
    return html.Div(
        className="chart-card",
        children=[dcc.Graph(id=graph_id, figure=figure, config={"displayModeBar": False, "responsive": True})],
    )


def build_charts_section() -> html.Div:
    # Create the grid layout containing both bar and pie charts
    """Create the bar chart and pie chart section."""
    return html.Div(
        className="charts-grid",
        children=[build_chart_card("bar-chart", empty_figure()), build_chart_card("pie-chart", empty_figure())],
    )


def build_footer() -> html.Div:
    # Create the footer with project credits
    """Create the footer shown at the bottom of the dashboard."""
    return html.Div(
        className="footer",
        children=[
            html.Div(
                className="footer-text",
                children=[
                    html.Span("CNN Text Classification Dashboard"),
                    html.Span("•"),
                    html.Span("Character-level Model"),
                    html.Span("•"),
                    html.Span("AG News Dataset"),
                ],
            ),
        ],
    )


def build_main_content() -> html.Div:
    # Create the main content area containing header, input, results, and charts
    """Create the main content area on the right side of the dashboard."""
    return html.Div(
        className="main-content",
        children=[
            html.Div(
                className="header",
                children=[
                    html.H1("CNN Text Classification", className="page-title"),
                    html.P("Enter text to classify it into News categories using character-level CNN", className="page-subtitle"),
                ],
            ),
            build_input_card(),
            dcc.Loading(
                id="loading",
                type="circle",
                color=THEME["accent"],
                children=html.Div(
                    id="results-container",
                    children=[
                        html.Div(className="card", style=CARD_STYLE, children=[html.Div("Prediction Result", className="card-title"), build_prediction_summary()]),
                        build_probability_section(),
                        build_charts_section(),
                    ],
                ),
            ),
            build_footer(),
        ],
    )


def build_layout() -> html.Div:
    # Assemble the complete page layout (sidebar + main content)
    """Assemble the full page layout from logical sections."""
    return html.Div(className="container", children=[build_sidebar(), build_main_content()])


# =============================
# Helper: Placeholder Figures

def empty_figure(message: str = "Results appear after prediction") -> go.Figure:
    """Create a dark-themed placeholder chart."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 13, "color": THEME["muted"]},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        # Create a dark-themed placeholder chart shown before prediction
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=320,
    )
    return fig


# =============================
# Helper: Prediction

def normalize_prediction(raw_result: dict) -> Tuple[str, float, Dict[str, float]]:
    # Parse and standardize prediction output from the model
    """Normalize predictor output into (label, confidence, probabilities)."""
    if not isinstance(raw_result, dict):
        raise ValueError("Predict returned non-dictionary.")

    if "probabilities" in raw_result:
        probs = raw_result.get("probabilities", {})
        if not isinstance(probs, dict) or not probs:
            raise ValueError("Missing probabilities.")
        clean_probs = {str(k): float(v) for k, v in probs.items()}
        label = str(raw_result.get("label") or max(clean_probs, key=clean_probs.get))
        confidence = float(raw_result.get("confidence", clean_probs.get(label, 0.0)))
        return label, confidence, clean_probs

    # Alternative format: direct class->prob dict
    clean_probs = {str(k): float(v) for k, v in raw_result.items()}
    if not clean_probs:
        raise ValueError("Empty prediction.")
    label = max(clean_probs, key=clean_probs.get)
    confidence = float(clean_probs[label])
    return label, confidence, clean_probs


def run_prediction(text: str) -> Tuple[bool, str, float, Dict[str, float], str]:
    # Safe wrapper for model prediction with input validation and error handling
    """Run prediction safely and return (success, label, confidence, probs, message)."""
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return False, "No Input", 0.0, {}, "Please enter text."

    if not os.path.exists(MODEL_PATH):
        return False, "Model Missing", 0.0, {}, f"Model not found: {MODEL_PATH}"

    try:
        signature = inspect.signature(predict)
        if len(signature.parameters) == 1:
            raw = predict(cleaned_text)
        else:
            raw = predict(cleaned_text, MODEL_PATH, CONFIG_NAME)

        if isinstance(raw, str):
            return False, "Prediction Error", 0.0, {}, raw

        label, confidence, probs = normalize_prediction(raw)
        return True, label, confidence, probs, "Success"

    except Exception as exc:
        return False, "Prediction Error", 0.0, {}, f"Error: {exc}"


# =============================
# Helper: Chart Builders

def build_bar_chart(probabilities: Dict[str, float], predicted_label: str) -> go.Figure:
    # Create horizontal bar chart with predicted class highlighted
    """Build horizontal bar chart highlighting predicted class."""
    if not probabilities:
        return empty_figure("No probability data")

    labels = list(probabilities.keys())
    values = [max(0.0, min(1.0, float(probabilities[l]))) for l in labels]
    colors = [THEME["accent"] if l == predicted_label else THEME["panel_light"] for l in labels]

    fig = go.Figure(
        data=[
            go.Bar(
                y=labels,
                x=values,
                orientation="h",
                marker=dict(color=colors, line=dict(color=THEME["border"], width=1)),
                text=[f"{v * 100:.1f}%" for v in values],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Probability: %{x:.2%}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=80, r=40, t=30, b=40),
        font=dict(color=THEME["text"], size=12),
        title=dict(text="Bar Chart", x=0.02, xanchor="left", font={"size": 14}),
        xaxis=dict(title="Probability", range=[0, 1], tickformat=".0%", gridcolor=f"rgba(255,255,255,0.05)"),
        yaxis=dict(title="Class"),
        height=280,
    )
    return fig


def build_pie_chart(probabilities: Dict[str, float], predicted_label: str) -> go.Figure:
    # Create pie chart showing distribution of class probabilities
    """Build pie chart showing class probability distribution."""
    if not probabilities:
        return empty_figure("No probability data")

    labels = list(probabilities.keys())
    values = [max(0.0, min(1.0, float(probabilities[l]))) for l in labels]
    colors = [THEME["accent"] if l == predicted_label else THEME["panel_light"] for l in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors, line=dict(color=THEME["panel"], width=2)),
                textinfo="label+percent",
                textfont=dict(color=THEME["text"], size=11),
                hovertemplate="<b>%{label}</b><br>Probability: %{value:.2%}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(color=THEME["text"], size=12),
        title=dict(text="Pie Chart", x=0.02, xanchor="left", font={"size": 14}),
        height=280,
    )
    return fig


def build_probability_cards(probabilities: Dict[str, float], predicted_label: str) -> List[html.Div]:
    # Create grid of probability cards sorted by confidence
    """Create probability cards for each class."""
    if not probabilities:
        return [build_empty_probability_card(class_name) for class_name in DEFAULT_CLASS_ORDER]

    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    return [build_probability_card(class_name, prob, predicted_label) for class_name, prob in sorted_items]


# ======================
# App Initialization

app = Dash(__name__, title="CNN Text Classification Dashboard", suppress_callback_exceptions=True)
server = app.server

# Custom HTML template with enhanced styling
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
            * { 
                margin: 0; 
                padding: 0; 
                box-sizing: border-box; 
            }
            
            body {
                font-family: 'Inter', sans-serif;
                background-color: #0d1117;
                color: #e6edf3;
                line-height: 1.5;
            }
            
            .container {
                display: flex;
                min-height: 100vh;
            }
            
            .sidebar {
                width: 260px;
                background-color: #161b22;
                border-right: 1px solid #30363d;
                padding: 20px;
                overflow-y: auto;
                box-shadow: 2px 0 8px rgba(0, 0, 0, 0.3);
            }
            
            .sidebar-section {
                margin-bottom: 24px;
            }
            
            .sidebar-title {
                font-size: 12px;
                font-weight: 700;
                color: #9aa4b2;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 12px;
            }
            
            .sidebar-item {
                display: flex;
                justify-content: space-between;
                padding: 10px;
                background-color: #0d1117;
                border-radius: 6px;
                margin-bottom: 8px;
                font-size: 13px;
                border: 1px solid #30363d;
            }
            
            .sidebar-label {
                color: #9aa4b2;
                font-weight: 500;
            }
            
            .sidebar-value {
                color: #63a4ff;
                font-weight: 600;
            }
            
            .example-btn {
                display: block;
                width: 100%;
                padding: 10px;
                margin-bottom: 8px;
                background-color: #238636;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                cursor: pointer;
                transition: all 200ms ease;
                text-align: left;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            
            .example-btn:hover {
                background-color: #2ea043;
                transform: translateX(2px);
                box-shadow: 0 4px 12px rgba(56, 211, 112, 0.3);
            }
            
            .example-btn:active {
                transform: translateX(0px);
            }
            
            .main-content {
                flex: 1;
                padding: 28px;
                overflow-y: auto;
                background: linear-gradient(135deg, rgba(13, 17, 23, 0.98), rgba(22, 27, 34, 0.95));
            }
            
            .header {
                margin-bottom: 28px;
            }
            
            .page-title {
                font-size: 32px;
                font-weight: 700;
                margin-bottom: 8px;
                background: linear-gradient(135deg, #63a4ff, #8bc0ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .page-subtitle {
                font-size: 14px;
                color: #9aa4b2;
                margin-bottom: 4px;
            }
            
            .card {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
                transition: all 200ms ease;
            }
            
            .card:hover {
                border-color: #63a4ff;
                box-shadow: 0 8px 16px rgba(88, 166, 255, 0.15);
                transform: translateY(-2px);
            }
            
            .card-title {
                font-size: 16px;
                font-weight: 700;
                margin-bottom: 14px;
                display: flex;
                align-items: center;
            }
            
            .textarea {
                width: 100%;
                min-height: 150px;
                padding: 14px;
                background-color: #0d1117;
                color: #e6edf3;
                border: 1px solid #30363d;
                border-radius: 8px;
                font-family: 'Inter', sans-serif;
                font-size: 14px;
                resize: vertical;
                transition: all 200ms ease;
            }
            
            .textarea:focus {
                outline: none;
                border-color: #63a4ff;
                box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.1);
            }
            
            .predict-btn {
                width: 100%;
                padding: 12px;
                margin-top: 12px;
                background: linear-gradient(135deg, #238636, #2ea043);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 700;
                cursor: pointer;
                transition: all 200ms ease;
            }
            
            .predict-btn:hover {
                background: linear-gradient(135deg, #2ea043, #3fb950);
                transform: translateY(-2px);
                box-shadow: 0 8px 16px rgba(56, 211, 112, 0.3);
            }
            
            .predict-btn:active {
                transform: translateY(0px);
            }
            
            .grid-2 {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin-top: 14px;
            }
            
            .metric-box {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 12px;
                text-align: center;
            }
            
            .metric-label {
                font-size: 12px;
                color: #9aa4b2;
                margin-bottom: 8px;
                font-weight: 600;
                text-transform: uppercase;
            }
            
            .metric-value {
                font-size: 24px;
                font-weight: 700;
                color: #63a4ff;
            }
            
            .confidence-meter {
                width: 100%;
                height: 10px;
                background-color: #0d1117;
                border-radius: 999px;
                overflow: hidden;
                border: 1px solid #30363d;
                margin-top: 8px;
            }
            
            .confidence-bar {
                height: 100%;
                width: 0%;
                background: linear-gradient(90deg, #3fb950, #238636);
                transition: width 500ms ease;
                border-radius: 999px;
            }
            
            .confidence-label {
                font-size: 11px;
                color: #9aa4b2;
                margin-top: 6px;
                font-weight: 500;
            }
            
            .charts-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            
            .chart-card {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 16px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            }
            
            .footer {
                background-color: #161b22;
                border-top: 1px solid #30363d;
                padding: 16px;
                text-align: center;
                color: #8b949e;
                font-size: 12px;
                margin-top: 28px;
            }
            
            .footer-text {
                display: flex;
                justify-content: center;
                gap: 8px;
                flex-wrap: wrap;
            }
            
            @media (max-width: 768px) {
                .sidebar {
                    width: 100%;
                    border-right: none;
                    border-bottom: 1px solid #30363d;
                    padding: 16px;
                }
                
                .container {
                    flex-direction: column;
                }
                
                .main-content {
                    padding: 16px;
                }
                
                .page-title {
                    font-size: 24px;
                }
                
                .charts-grid {
                    grid-template-columns: 1fr;
                }
                
                .grid-2 {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


# Layout


app.layout = build_layout()


# =============================
# Callbacks

# Callback: Load example text when example buttons are clicked
@app.callback(
    Output("input-text", "value"),
    [
        Input("example-btn-sports", "n_clicks"),
        Input("example-btn-business", "n_clicks"),
        Input("example-btn-scitech", "n_clicks"),
        Input("example-btn-world", "n_clicks"),
    ],
)
def load_example_text(sports_clicks, business_clicks, scitech_clicks, world_clicks):
    # Load example text into input field when corresponding button is clicked
    """Load example text when user clicks example buttons.
    
    This callback listens to all example buttons and loads the corresponding
    example text into the input textarea when clicked. The callback determines
    which button was most recently clicked by comparing click counts.
    """
    # Determine which button was most recently clicked
    buttons = {
        "sports": (sports_clicks, EXAMPLE_TEXTS["Sports"]),
        "business": (business_clicks, EXAMPLE_TEXTS["Business"]),
        "scitech": (scitech_clicks, EXAMPLE_TEXTS["Sci/Tech"]),
        "world": (world_clicks, EXAMPLE_TEXTS["world"]),
    }
    
    max_clicks = max([count for count, _ in buttons.values()])
    if max_clicks == 0:
        return ""
    
    for name, (count, text) in buttons.items():
        if count == max_clicks:
            return text
    
    return ""


# Callback: Main prediction and UI
@app.callback(
    Output("predicted-class", "children"),
    Output("prediction-desc", "children"),
    Output("confidence-text", "children"),
    Output("confidence-badge", "children"),
    Output("confidence-badge", "style"),
    Output("confidence-bar", "style"),
    Output("confidence-label", "children"),
    Output("probability-cards", "children"),
    Output("bar-chart", "figure"),
    Output("pie-chart", "figure"),
    Output("status-message", "children"),
    Input("predict-btn", "n_clicks"),
    State("input-text", "value"),
    prevent_initial_call=True,
)
def update_results(n_clicks, text_input):
    # Main callback: predict text classification and update all UI components
    """Main callback: run prediction and update all dashboard components.
    
    Interaction flow:
    1. User enters text and clicks Predict button
    2. run_prediction() is called which:
       - Validates input
       - Calls Predict.predict() for model inference
       - Gets softmax probabilities from the model
    3. normalize_prediction() extracts label, confidence, and class probabilities
    4. Helper functions build all UI components (cards, charts) 
    5. Confidence color indicator changes based on confidence level:
       - Green (high): >= 70%
       - Yellow (medium): 40-70%
       - Red (low): < 40%
    6. All outputs are updated atomically for a synchronized UI
    
    Returns:
        Tuple of output values for all Output decorators in order.
    """
    success, label, confidence, probabilities, message = run_prediction(text_input)
    
    if not success:
        # Error state: show red confidence indicator
        error_badge_style = build_confidence_badge_style(THEME["low_conf"])
        error_bar_style = build_confidence_bar_style(THEME["low_conf"], "0%")
        return (
            label,
            message,
            "0.00%",
            "Error",
            error_badge_style,
            error_bar_style,
            "0%",
            build_probability_cards({}, ""),
            empty_figure("Prediction failed"),
            empty_figure("Prediction failed"),
            message,
        )
    
    # Success state: determine confidence color and update UI
    confidence_pct = max(0.0, min(1.0, confidence)) * 100
    conf_color = get_confidence_color(confidence)
    conf_label = get_confidence_label(confidence)

    success_badge_style = build_confidence_badge_style(conf_color)
    success_bar_style = build_confidence_bar_style(conf_color, f"{confidence_pct:.1f}%")
    
    description = f"Successfully classified as '{label}' with {confidence_pct:.1f}% confidence."
    
    return (
        label,
        description,
        f"{confidence_pct:.1f}%",
        conf_label,
        success_badge_style,
        success_bar_style,
        f"{confidence_pct:.0f}%",
        build_probability_cards(probabilities, label),
        build_bar_chart(probabilities, label),
        build_pie_chart(probabilities, label),
        "Prediction completed ",
    )


# Entrypoint

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=7860, debug=False)
