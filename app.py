import streamlit as st
import plotly.graph_objects as go
import numpy as np
import math

from tokenizer import Tokenizer
from geometry import (
    get_semantic,
    sinusodial_encoding,
    compute_additive,
    simulate_rope_rotation
)

st.set_page_config(layout="wide")

st.title("TesseraAxis: The Geometry of Sequence")
st.write("Deconstructing Tokenization and High-Dimensional Vector Spaces from First Principles.")

training_corpus = (
    "Behold the machine. The mathematical architecture of the universe operates "
    "on pure, logical dynamics. Inevitability rules the latent spaces of deep learning."
)

@st.cache_resource
def initialize_and_train_tokenizer():
    # Allow 24 custom merges above the 256 baseline bytes
    tok = Tokenizer(vocab_size=280)
    tok.train(training_corpus)
    tok.save("deepseek_bpe")
    return tok

tokenizer = initialize_and_train_tokenizer()


user_text = st.text_input(
    "Enter a custom sentence to pass through the model pipeline:", 
    value="Inevitability rules."
)
token_ids = tokenizer.encode(user_text)
visual_tokens = tokenizer.get_visual_tokens(token_ids)
if not token_ids:
    st.warning("Please enter a string of characters to begin analysis.")
    st.stop()

token_embeddings = get_semantic(token_ids, d_model=16)
pos_embeddings = sinusodial_encoding(len(token_ids), d_model=16)
additive_final = compute_additive(token_embeddings, pos_embeddings)

st.write("### Stage 1: The Tokenized Registry")
st.write("The sentence is isolated by your regex rules and compressed into vocabulary indices.")
columns = st.columns(len(visual_tokens))
for idx, (token_string, tid) in enumerate(zip(visual_tokens, token_ids)):
    with columns[idx]:
        st.metric(
            label=f"Sequence Pos {idx}", 
            value=f"'{token_string}'", 
            delta=f"ID: {tid}",
            delta_color="off"
        )
st.write("## Stage 2: The Behind-the-Scenes Numerical Ledger")
st.write("Select a token from your sentence to open up the model's memory banks and watch the raw math")
selected_pos = st.selectbox(
    "Choose a Token Position to Inspect:",
    options=list(range(len(visual_tokens))),
    format_func=lambda i: f"Position {i}: '{visual_tokens[i]}' (ID: {token_ids[i]})"
)
t_vec = token_embeddings[selected_pos]
p_vec = pos_embeddings[selected_pos]
a_vec = additive_final[selected_pos]
st.markdown(f"### Currently Inspecting: **'{visual_tokens[selected_pos]}'** at Position **{selected_pos}**")
st.write("#### ──► Path A Math: Word Vector + Position Vector (Element-wise Sum)")
math_ledger_data = {
    "Dimension Coordinate": [f"Dimension {i}" for i in range(4)],
    "Base Word Vector (Semantic)": [f"{t_vec[i]:.4f}" for i in range(4)],
    "Positional Wave Vector (Time)": [f"{p_vec[i]:.4f}" for i in range(4)],
    "Final Superposition Vector": [f"{a_vec[i]:.4f}" for i in range(4)]
}
st.table(math_ledger_data)
st.caption(
    "Notice the literal math: for every individual coordinate dimension, the Base Word value "
    "and the Positional Wave value are added together. This directly alters the core coordinates of the token."
)

st.write("#### ──► Path B Math: Rotary Coordinates (Fixed Magnitude, Angular Shift)")
simulated_timeline = list(range(11))
rope_points = simulate_rope_rotation(t_vec, simulated_timeline, theta_base=0.4)
fig_rope = go.Figure()
vector_magnitude = math.sqrt(t_vec[0]**2 + t_vec[1]**2)
angles = np.linspace(0, 2 * np.pi, 100)
fig_rope.add_trace(go.Scatter(
    x=vector_magnitude * np.cos(angles), 
    y=vector_magnitude * np.sin(angles),
    mode='lines', 
    line=dict(color='rgba(200,200,200,0.4)', dash='dash'),
    name="Constant Magnitude Orbit", 
    showlegend=True
))
for pt in rope_points:
    p = pt["position"]
    x_val = pt["x"]
    y_val = pt["y"]
    is_current = (p == selected_pos)
    line_color = "crimson" if is_current else "rgba(100,149,237,0.6)"
    line_width = 4 if is_current else 1.5
    fig_rope.add_trace(go.Scatter(
        x=[0, x_val], 
        y=[0, y_val],
        mode='lines+markers',
        name=f"If at Pos {p}" + (" (Current)" if is_current else ""),
        line=dict(color=line_color, width=line_width),
        marker=dict(size=5)
    ))

fig_rope.update_layout(
    title=f"Rotary Orbit for '{visual_tokens[selected_pos]}' on the 2D Plane (Dimensions d0 & d1)",
    xaxis=dict(title="Dimension 0 Coordinate Axis", range=[-1.5, 1.5], zeroline=True, zerolinecolor="gray"),
    yaxis=dict(
        title="Dimension 1 Coordinate Axis", 
        range=[-1.5, 1.5], 
        zeroline=True, 
        zerolinecolor="gray",
        scaleanchor="x",  # ◄─── Shifted inside yaxis
        scaleratio=1      # ◄─── Shifted inside yaxis
    ),
    width=700, 
    height=600
    # ◄─── Removed from global layout level
) 
col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(fig_rope, use_container_width=False)
with col2:
    st.write("#### Mechanical Observations:")
    st.markdown(
        f"* **The Crimson Vector:** This shows exactly where the coordinates for **'{visual_tokens[selected_pos]}'** "
        f"point right now because it sits at Position **{selected_pos}** in your input field.\n\n"
        f"* **The Blue Variants:** These illustrate the alternative states if the same word shifted order. "
        f"Notice they stay perfectly fixed to the circular orbit line.\n\n"
        f"* **No Contamination:** Because length is conserved, the underlying semantic definition of the token remains completely "
        f"unaltered before it enters DeepSeek's low-rank Multi-head Latent Attention (MLA) layers."
    )