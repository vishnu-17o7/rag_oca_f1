"""
Streamlit UI for RAG Hyperparameter Optimization
Monochrome v0-inspired design
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Page config
st.set_page_config(
    page_title="RAG Optimizer",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for v0-inspired monochrome theme
st.markdown(
    """
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #ffffff !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #0a0a0a;
        border: 1px solid #333333;
        color: #ffffff;
        border-radius: 6px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #666666;
        outline: none;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #ffffff;
        color: #000000;
        border: none;
        border-radius: 6px;
        padding: 12px 24px;
        font-weight: 500;
        font-size: 14px;
        width: auto;
        display: block;
        margin: 0 auto;
    }
    
    .stButton > button:hover {
        background-color: #e0e0e0;
    }
    
    .stButton > button[kind="secondary"] {
        background-color: transparent;
        border: 1px solid #333333;
        color: #ffffff;
    }
    
    .stButton > button[kind="secondary"]:hover {
        border-color: #666666;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #ffffff;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: 1px solid #333333;
        border-radius: 6px;
        color: #888888;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #000000;
    }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #0a0a0a;
        border: 1px solid #222222;
        border-radius: 8px;
        padding: 16px;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #666666 !important;
        font-size: 12px;
    }
    
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 24px;
        font-weight: 600;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #0a0a0a;
        border: 1px solid #222222;
        border-radius: 6px;
        color: #888888 !important;
    }
    
    /* Dataframe */
    div[data-testid="stDataFrame"] {
        background-color: #0a0a0a;
        border: 1px solid #222222;
        border-radius: 8px;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        max-width: 85%;
        word-wrap: break-word;
    }
    
    .chat-user {
        background-color: #111111;
        border: 1px solid #222222;
        margin-left: auto;
    }
    
    .chat-assistant {
        background-color: #0a0a0a;
        border: 1px solid #222222;
        margin-right: auto;
    }
    
    .chat-label {
        font-size: 11px;
        color: #666666;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Separator */
    hr {
        border-color: #222222;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #333333;
        border-radius: 4px;
    }
    
    /* Centered container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }
    
    /* Title styling */
    .title-text {
        font-size: 28px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 8px;
    }
    
    .subtitle-text {
        font-size: 14px;
        color: #666666;
        text-align: center;
        margin-bottom: 32px;
    }
    
    /* Tab container */
    .tabs-container {
        margin-top: 24px;
    }
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "best_params" not in st.session_state:
        st.session_state.best_params = None
    if "best_score" not in st.session_state:
        st.session_state.best_score = 0.0
    if "running" not in st.session_state:
        st.session_state.running = False
    if "complete" not in st.session_state:
        st.session_state.complete = False
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    if "optimization_key" not in st.session_state:
        st.session_state.optimization_key = 0


def run_optimization():
    """Run the optimization and update session state."""
    import numpy as np
    from research.oca import OverclockingAlgorithm
    from src.fitness import fitness_function

    BOUNDS = [
        (100, 1000),
        (0, 200),
        (0.0, 1.0),
        (1, 10),
    ]
    MAX_EVALUATIONS = 50
    POP_SIZE = 10
    NUM_P_CORES = 3
    INITIAL_VOLTAGE = 2.0
    FINAL_VOLTAGE = 0.0

    class OCAOptimizer:
        def __init__(self, bounds, max_evals, pop_size, num_p_cores):
            self.bounds = bounds
            self.max_evals = max_evals
            self.dim = len(bounds)
            self.oca = OverclockingAlgorithm(
                pop_size=pop_size,
                num_p_cores=num_p_cores,
                initial_voltage=INITIAL_VOLTAGE,
                final_voltage=FINAL_VOLTAGE,
                aggressive_voltage=True,
            )
            self.evaluation_count = 0
            self.history = []
            self.best_fitness = -np.inf
            self.best_params = None

        def _clip_to_bounds(self, position):
            clipped = np.zeros(self.dim)
            for i, (lower, upper) in enumerate(self.bounds):
                clipped[i] = np.clip(position[i], lower, upper)
            return clipped

        def _objective_wrapper(self, position):
            clipped = self._clip_to_bounds(position)
            fitness = fitness_function(clipped.tolist())
            self.evaluation_count += 1

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_params = clipped.copy()

            iteration = self.evaluation_count - 1
            self.history.append((iteration, self.best_fitness, self.best_params.copy()))

            st.session_state.history = self.history.copy()
            st.session_state.best_params = self.best_params.copy()
            st.session_state.best_score = self.best_fitness

            return fitness

        def run(self):
            max_iterations = max(1, self.max_evals // POP_SIZE)
            all_lower = min(b[0] for b in self.bounds)
            all_upper = max(b[1] for b in self.bounds)

            self.oca.optimize(
                objective_fn=self._objective_wrapper,
                bounds=(all_lower, all_upper),
                dim=self.dim,
                max_iterations=max_iterations,
            )

            remaining = self.max_evals - self.evaluation_count
            while remaining > 0 and self.evaluation_count < self.max_evals:
                for _ in range(min(remaining, POP_SIZE)):
                    random_pos = np.array(
                        [np.random.uniform(b[0], b[1]) for b in self.bounds]
                    )
                    self._objective_wrapper(random_pos)
                    remaining = self.max_evals - self.evaluation_count
                    if remaining <= 0:
                        break

            return self.best_params, self.best_fitness, self.history

    optimizer = OCAOptimizer(BOUNDS, MAX_EVALUATIONS, POP_SIZE, NUM_P_CORES)
    best_params, best_score, history = optimizer.run()

    st.session_state.running = False
    st.session_state.complete = True


def plot_convergence_streamlit(history):
    """Generate convergence plot for Streamlit."""
    if not history:
        return None

    iterations = [h[0] + 1 for h in history]
    fitness_scores = [h[1] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")

    ax.plot(
        iterations, fitness_scores, color="#ffffff", linewidth=2, label="Best Fitness"
    )
    ax.axhline(
        y=0.8, color="#666666", linestyle="--", linewidth=1.5, label="Target (0.8)"
    )

    ax.set_xlabel("Iteration", color="#666666", fontsize=11)
    ax.set_ylabel("Fitness Score", color="#666666", fontsize=11)
    ax.set_title("Convergence", color="#ffffff", fontsize=13, fontweight="500")
    ax.set_xlim(1, max(iterations))
    ax.set_ylim(0, 1)
    ax.tick_params(colors="#666666", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.legend(loc="lower right", facecolor="#0a0a0a", labelcolor="#888888", fontsize=10)
    ax.grid(True, alpha=0.15, color="#333333")

    return fig


def create_rag_pipeline():
    """Create RAG pipeline with best params."""
    if st.session_state.best_params is None:
        return None

    from src.rag_pipeline import RAGPipeline

    params = st.session_state.best_params
    return RAGPipeline(
        chunk_size=int(round(params[0])),
        chunk_overlap=int(round(params[1])),
        top_k=int(round(params[3])),
        temperature=float(params[2]),
    )


def render_chat_message(role, content):
    """Render a chat message with custom styling."""
    label = "You" if role == "user" else "Assistant"
    css_class = "chat-user" if role == "user" else "chat-assistant"

    st.markdown(
        f"""
    <div class="chat-message {css_class}">
        <div class="chat-label">{label}</div>
        {content}
    </div>
    """,
        unsafe_allow_html=True,
    )


def main():
    init_session_state()

    # Header
    st.markdown(
        '<div class="title-text">RAG Hyperparameter Optimizer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle-text">Optimize chunk_size, chunk_overlap, temperature, and top_k using OCA</div>',
        unsafe_allow_html=True,
    )

    # Tabs
    tab1, tab2 = st.tabs(["Optimize", "Chat"])

    with tab1:
        # Start button
        if not st.session_state.running and not st.session_state.complete:
            if st.button("Start Optimization"):
                st.session_state.running = True
                st.session_state.history = []
                st.session_state.best_params = None
                st.session_state.best_score = 0.0
                st.session_state.complete = False
                run_optimization()
                st.rerun()

        # Show progress/results
        if st.session_state.running:
            st.markdown("### Running...")
            progress = len(st.session_state.history) / 50
            st.progress(min(progress, 1.0))
            st.markdown(f"**{len(st.session_state.history)} / 50 evaluations**")

        if st.session_state.history:
            # Best results
            st.markdown("---")
            st.markdown("### Best Results")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Fitness", f"{st.session_state.best_score:.4f}")
            with col2:
                st.metric(
                    "chunk_size", f"{int(round(st.session_state.best_params[0]))}"
                )
            with col3:
                st.metric(
                    "chunk_overlap", f"{int(round(st.session_state.best_params[1]))}"
                )
            with col4:
                st.metric("temperature", f"{st.session_state.best_params[2]:.2f}")
            with col5:
                st.metric("top_k", f"{int(round(st.session_state.best_params[3]))}")

            # Convergence plot
            fig = plot_convergence_streamlit(st.session_state.history)
            if fig:
                st.pyplot(fig)

            # History table
            if len(st.session_state.history) > 0:
                with st.expander("History"):
                    data = []
                    for iteration, best_fitness, params in st.session_state.history:
                        data.append(
                            {
                                "#": iteration + 1,
                                "Fitness": f"{best_fitness:.4f}",
                                "chunk_size": int(round(params[0])),
                                "chunk_overlap": int(round(params[1])),
                                "temperature": f"{params[2]:.2f}",
                                "top_k": int(round(params[3])),
                            }
                        )
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True, hide_index=True)

        if st.session_state.complete:
            st.markdown("---")
            st.markdown("**Optimization complete**")

            if st.button("Run Again"):
                st.session_state.running = False
                st.session_state.complete = False
                st.session_state.history = []
                st.session_state.best_params = None
                st.session_state.best_score = 0.0
                st.session_state.optimization_key += 1
                st.rerun()

    with tab2:
        st.markdown("### Chat")

        # Check if optimization is complete
        if not st.session_state.complete or st.session_state.best_params is None:
            st.markdown("Optimization not yet complete. Chat uses default parameters.")
            # Allow chat with default params even without optimization
            if st.session_state.rag_pipeline is None:
                with st.spinner("Loading RAG pipeline with defaults..."):
                    from src.rag_pipeline import RAGPipeline
                    st.session_state.rag_pipeline = RAGPipeline(
                        chunk_size=700,
                        chunk_overlap=100,
                        top_k=4,
                        temperature=0.0,
                    )
        else:
            # Initialize RAG pipeline if needed
            if st.session_state.rag_pipeline is None:
                with st.spinner("Loading RAG pipeline..."):
                    st.session_state.rag_pipeline = create_rag_pipeline()

            # Display chat messages
            for msg in st.session_state.chat_messages:
                render_chat_message(msg["role"], msg["content"])

            # Chat input
            with st.form(key="chat_form", clear_on_submit=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    user_input = st.text_input(
                        "Ask a question",
                        placeholder="Type your question...",
                        label_visibility="collapsed",
                    )
                with col2:
                    submit = st.form_submit_button("Send")

            if submit and user_input:
                # Add user message
                st.session_state.chat_messages.append(
                    {"role": "user", "content": user_input}
                )

                # Get response
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_pipeline.query(user_input)
                    except Exception as e:
                        response = f"Error: {str(e)}"

                # Add assistant message
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": response}
                )
                st.rerun()

            # Clear chat button
            if st.session_state.chat_messages:
                if st.button("Clear Chat"):
                    st.session_state.chat_messages = []
                    st.rerun()


if __name__ == "__main__":
    main()
