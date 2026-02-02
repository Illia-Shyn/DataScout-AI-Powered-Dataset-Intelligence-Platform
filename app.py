import streamlit as st
import pandas as pd
import time

from components.data_profiler import DataProfiler
from components.visualizer import AutoVisualizer
from components.anomaly_detector import AnomalyDetector
from components.llm_chat import DataChat

# Page Configuration
st.set_page_config(
    page_title="DataScout - AI Dataset Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    local_css("assets/style.css")
except:
    pass # CSS file might not exist yet during testing

# --- Session State Initialization ---
if 'data_profile' not in st.session_state:
    st.session_state.data_profile = None
if 'column_types' not in st.session_state:
    st.session_state.column_types = None
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Caching Expensive Operations ---
@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        # Retry with a different encoding if utf-8 fails
        file.seek(0)
        return pd.read_csv(file, encoding='latin1')

@st.cache_data
def generate_profile(df):
    profiler = DataProfiler(df)
    return profiler.generate_profile(), profiler.detect_column_types()

@st.cache_data
def detect_anomalies(df, contamination):
    detector = AnomalyDetector(df)
    return detector.detect_all_anomalies(contamination=contamination)

@st.cache_data
def generate_charts(df, column_types):
    visualizer = AutoVisualizer(df, column_types)
    return visualizer.generate_all_charts()

# --- Main App ---
def main():
    st.title("DataScout 🔍")
    st.caption("AI-Powered Dataset Intelligence Platform")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")
        
        st.divider()
        st.subheader("Settings")
        contamination = st.slider("Anomaly Contamination", 0.01, 0.20, 0.05, 0.01)
        model_name = st.text_input("Ollama Model", value="gemma3:4b")
        
        st.info(f"Using model: {model_name}")

    if uploaded_file is None:
        st.info("👋 Welcome! Please upload a CSV file in the sidebar to begin analysis.")
        return

    # Load Data
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return

    # --- Analysis Pipeline ---
    # 1. Profile Data
    if st.session_state.data_profile is None or uploaded_file.name != st.session_state.get('current_file'):
        with st.spinner("Profiling data..."):
            profile, col_types = generate_profile(df)
            st.session_state.data_profile = profile
            st.session_state.column_types = col_types
            st.session_state.current_file = uploaded_file.name
            # Reset other states
            st.session_state.anomalies = None
            st.session_state.chat_history = []

    profile = st.session_state.data_profile
    col_types = st.session_state.column_types

    # 2. Tabs Interface
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Profiling", "📈 Patterns", "🚨 Anomalies", "💬 AI Chat"])

    # --- TAB 1: Profiling ---
    with tab1:
        st.subheader("Dataset Overview")
        ov = profile['overview']
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", ov['n_rows'])
        m2.metric("Columns", ov['n_cols'])
        m3.metric("Missing Values", ov['total_missing'])
        m4.metric("Duplicates", ov['duplicates'])

        st.subheader("Sample Data")
        st.dataframe(df.head())

        st.subheader("Column Details")
        cols_df = pd.DataFrame.from_dict(profile['columns'], orient='index')
        st.dataframe(cols_df, use_container_width=True)
    
    # --- TAB 2: Visualizations ---
    with tab2:
        st.subheader("Smart Visualizations")
        charts = generate_charts(df, col_types)
        
        if not charts:
            st.warning("No suitable charts could be generated automatically.")
        else:
            # Display in grid
            c1, c2 = st.columns(2)
            for i, chart in enumerate(charts):
                with (c1 if i % 2 == 0 else c2):
                    st.plotly_chart(chart['figure'], use_container_width=True)
                    st.caption(f"💡 {chart['insight']}")

    # --- TAB 3: Anomalies ---
    with tab3:
        st.subheader("Anomaly Detection")
        
        if st.session_state.anomalies is None:
            with st.spinner("Detecting anomalies..."):
                anomalies = detect_anomalies(df, contamination)
                st.session_state.anomalies = anomalies
        else:
            anomalies = st.session_state.anomalies
        
        # Summary
        summary = anomalies['summary']
        st.metric("Total Anomalies Detected", summary['total_anomalies'], delta=f"{summary['anomaly_rate']:.1%} Rate", delta_color="inverse")
        
        # Details
        st.subheader("Anomalous Records")
        flagged_df = anomalies['flagged_rows']
        
        if not flagged_df.empty:
            # Reorder columns to put 'anomaly_reason' first or near valid data
            cols = ['anomaly_reason'] + [c for c in flagged_df.columns if c != 'anomaly_reason' and c != 'is_anomaly']
            st.dataframe(
                flagged_df[cols].style.apply(lambda x: ['background-color: #ffd7d7' for i in x], axis=1)
            )
        else:
            st.success("No significant anomalies detected.")

    # --- TAB 4: AI Chat ---
    with tab4:
        st.subheader("Chat with your Data")
        
        # Initialize Chat Interface
        chat_engine = DataChat(df, profile, anomalies, model_name=model_name)
        
        # Debug Context
        if st.checkbox("Show LLM Context (Debug)"):
            debug = chat_engine.debug_context()
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Chars:** {debug['char_count']}")
            c2.write(f"**~Tokens:** {debug['approx_tokens']}")
            c3.write(f"**Lines:** {debug['line_count']}")
            with st.expander("View Full Context"):
                st.code(debug['full_context'])

        # Container for chat history
        chat_container = st.container(height=500)
        
        # Display History
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        # Input
        if prompt := st.chat_input("Ask a question about your data..."):
            # 1. Add User Message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # 2. Add Assistant Message
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = chat_engine.ask(prompt)
                        answer = response['answer']
                        st.markdown(answer)
            
            st.session_state.chat_history.append({"role": "assistant", "content": answer})



if __name__ == "__main__":
    main()
