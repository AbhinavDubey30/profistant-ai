import streamlit as st
import requests
import json
import time
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="Profistant AI - Streamlit",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #667eea;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .search-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .paper-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sample papers for quick demo
SAMPLE_PAPERS = [
    {
        "title": "Recent Advances in Deep Learning",
        "authors": "Smith, J., Johnson, A., Williams, B.",
        "abstract": "This paper presents recent developments in deep learning architectures and their applications in various domains including computer vision, natural language processing, and robotics.",
        "url": "https://example.com/paper1",
        "year": "2023"
    },
    {
        "title": "Machine Learning in Healthcare",
        "authors": "Brown, C., Davis, M., Wilson, K.",
        "abstract": "We explore the application of machine learning techniques in healthcare, focusing on diagnostic tools, treatment optimization, and patient outcome prediction.",
        "url": "https://example.com/paper2",
        "year": "2023"
    },
    {
        "title": "Natural Language Processing Transformers",
        "authors": "Garcia, L., Martinez, P., Rodriguez, S.",
        "abstract": "This work examines the evolution of transformer architectures in NLP, from BERT to GPT models, and their impact on language understanding tasks.",
        "url": "https://example.com/paper3",
        "year": "2022"
    },
    {
        "title": "Computer Vision Applications",
        "authors": "Anderson, R., Taylor, E., Moore, F.",
        "abstract": "A comprehensive survey of computer vision applications in autonomous vehicles, medical imaging, and surveillance systems using deep learning approaches.",
        "url": "https://example.com/paper4",
        "year": "2023"
    },
    {
        "title": "Reinforcement Learning in Robotics",
        "authors": "Lee, H., Kim, S., Park, J.",
        "abstract": "This paper investigates reinforcement learning algorithms for robotic control, including policy gradient methods and their real-world implementations.",
        "url": "https://example.com/paper5",
        "year": "2023"
    }
]

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 Profistant AI - Streamlit Version</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Navigation")
        page = st.selectbox("Choose a page:", [
            "🔍 Research Search",
            "📖 Reading List", 
            "📅 Planner",
            "📊 Dashboard",
            "💡 Research Ideas"
        ])
        
        st.markdown("---")
        st.markdown("**Quick Demo Mode**")
        st.info("This Streamlit version provides instant results for demo purposes!")
    
    # Main content based on selected page
    if page == "🔍 Research Search":
        show_research_search()
    elif page == "📖 Reading List":
        show_reading_list()
    elif page == "📅 Planner":
        show_planner()
    elif page == "📊 Dashboard":
        show_dashboard()
    elif page == "💡 Research Ideas":
        show_research_ideas()

def show_research_search():
    st.header("🔍 Research Paper Search")
    
    # Search form
    with st.form("search_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            topic = st.text_input("Enter research topic:", placeholder="e.g., machine learning, deep learning, NLP")
        with col2:
            search_button = st.form_submit_button("🔍 Search", use_container_width=True)
    
    if search_button and topic:
        with st.spinner("Searching for papers..."):
            time.sleep(1)  # Simulate search time
            
            # Show sample papers (instant results)
            st.markdown('<div class="success-message">✅ Found 5 papers instantly! (Demo mode)</div>', unsafe_allow_html=True)
            
            for i, paper in enumerate(SAMPLE_PAPERS):
                with st.container():
                    st.markdown(f'<div class="paper-card">', unsafe_allow_html=True)
                    st.markdown(f"**{paper['title']}**")
                    st.markdown(f"*Authors:* {paper['authors']}")
                    st.markdown(f"*Year:* {paper['year']}")
                    st.markdown(f"*Abstract:* {paper['abstract']}")
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button(f"📖 Add to List", key=f"add_{i}"):
                            st.success(f"Added '{paper['title'][:30]}...' to reading list!")
                    with col2:
                        if st.button(f"📝 Summarize", key=f"sum_{i}"):
                            with st.spinner("Generating summary..."):
                                time.sleep(0.5)
                                st.info(f"**Summary:** This paper discusses {topic} and presents novel approaches in the field.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("---")

def show_reading_list():
    st.header("📖 Reading List")
    
    if 'reading_list' not in st.session_state:
        st.session_state.reading_list = []
    
    if st.session_state.reading_list:
        st.markdown(f"**You have {len(st.session_state.reading_list)} papers in your reading list:**")
        for i, paper in enumerate(st.session_state.reading_list):
            with st.expander(f"{paper['title'][:50]}..."):
                st.markdown(f"**Authors:** {paper['authors']}")
                st.markdown(f"**Abstract:** {paper['abstract']}")
                if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                    st.session_state.reading_list.pop(i)
                    st.rerun()
    else:
        st.info("Your reading list is empty. Add some papers from the Research Search page!")

def show_planner():
    st.header("📅 Research Planner")
    
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []
    
    # Add new task
    with st.form("add_task"):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_task = st.text_input("Add a new research task:", placeholder="e.g., Read paper on transformers, Implement model")
        with col2:
            add_button = st.form_submit_button("➕ Add")
        
        if add_button and new_task:
            st.session_state.tasks.append({
                'text': new_task,
                'completed': False,
                'date': datetime.now().strftime("%Y-%m-%d")
            })
            st.rerun()
    
    # Display tasks
    if st.session_state.tasks:
        st.markdown("**Your Research Tasks:**")
        for i, task in enumerate(st.session_state.tasks):
            col1, col2, col3 = st.columns([1, 6, 1])
            with col1:
                completed = st.checkbox("", value=task['completed'], key=f"task_{i}")
                if completed != task['completed']:
                    st.session_state.tasks[i]['completed'] = completed
                    st.rerun()
            with col2:
                if task['completed']:
                    st.markdown(f"~~{task['text']}~~ ✅")
                else:
                    st.markdown(task['text'])
            with col3:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.tasks.pop(i)
                    st.rerun()
    else:
        st.info("No tasks yet. Add your first research task above!")

def show_dashboard():
    st.header("📊 Research Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Papers Saved", len(st.session_state.get('reading_list', [])))
    
    with col2:
        total_tasks = len(st.session_state.get('tasks', []))
        completed_tasks = sum(1 for task in st.session_state.get('tasks', []) if task['completed'])
        st.metric("📅 Tasks", f"{completed_tasks}/{total_tasks}")
    
    with col3:
        st.metric("🎯 Progress", f"{int((completed_tasks/max(total_tasks,1))*100)}%")
    
    with col4:
        st.metric("⚡ Searches", "5")
    
    # Recent activity
    st.markdown("### 📈 Recent Activity")
    st.info("✅ Added 3 papers to reading list")
    st.info("📝 Generated 2 summaries")
    st.info("🔍 Completed 5 searches")

def show_research_ideas():
    st.header("💡 Research Ideas")
    
    if 'ideas' not in st.session_state:
        st.session_state.ideas = [
            {
                "title": "AI-Powered Research Assistant",
                "description": "Develop an intelligent system that can automatically analyze research papers and suggest relevant connections.",
                "category": "Artificial Intelligence",
                "likes": 15
            },
            {
                "title": "Blockchain for Academic Publishing", 
                "description": "Create a decentralized platform for academic publishing to ensure transparency and prevent plagiarism.",
                "category": "Blockchain",
                "likes": 8
            }
        ]
    
    # Add new idea
    with st.form("add_idea"):
        st.markdown("**Share Your Research Idea**")
        col1, col2 = st.columns(2)
        with col1:
            idea_title = st.text_input("Idea Title:")
        with col2:
            idea_category = st.text_input("Category:")
        
        idea_description = st.text_area("Description:")
        
        if st.form_submit_button("💡 Share Idea"):
            if idea_title and idea_description and idea_category:
                st.session_state.ideas.append({
                    "title": idea_title,
                    "description": idea_description,
                    "category": idea_category,
                    "likes": 0
                })
                st.rerun()
    
    # Display ideas
    st.markdown("### 🌟 Featured Research Ideas")
    for i, idea in enumerate(st.session_state.ideas):
        with st.expander(f"{idea['title']} ({idea['category']})"):
            st.markdown(f"**Description:** {idea['description']}")
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button(f"❤️ {idea['likes']}", key=f"like_{i}"):
                    st.session_state.ideas[i]['likes'] += 1
                    st.rerun()
            with col2:
                st.markdown(f"*Category: {idea['category']}*")

if __name__ == "__main__":
    main()
