import streamlit as st 
from scholarly import scholarly
from google import genai
import time
import random
from scholarly._navigator import MaxTriesExceededException

# Configuring gemini API
api_key = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

# Initialize scholarly with default settings
scholarly.set_timeout(15)

def search_papers_with_retry(topic, max_retries=3, timeout=15):
    """
    Search for papers with retry mechanism and error handling
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for attempt in range(max_retries):
        try:
            # Update progress
            progress = (attempt + 1) / max_retries
            progress_bar.progress(progress)
            status_text.text(f"Attempt {attempt + 1} of {max_retries}...")
            
            # Add random delay to avoid rate limiting
            if attempt > 0:
                delay = random.uniform(2, 5)
                status_text.text(f"Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
            
            # Configure scholarly with timeout
            scholarly.set_timeout(timeout)
            
            # Search for publications
            status_text.text("Searching Google Scholar...")
            search_results = scholarly.search_pubs(topic)
            
            # Success!
            progress_bar.progress(1.0)
            status_text.text("✅ Search successful!")
            time.sleep(1)  # Brief pause to show success
            progress_bar.empty()
            status_text.empty()
            
            return search_results
            
        except MaxTriesExceededException as e:
            st.warning(f"⚠️ Attempt {attempt + 1} failed: Google Scholar is blocking requests.")
            if attempt == max_retries - 1:
                progress_bar.empty()
                status_text.empty()
                st.error("❌ Unable to connect to Google Scholar after multiple attempts. This might be due to:")
                st.markdown("""
                - **Rate limiting**: Google Scholar has temporarily blocked requests
                - **Network issues**: Connection problems
                - **Proxy requirements**: Your network may require proxy configuration
                
                **Suggestions:**
                - Try again in a few minutes
                - Check your internet connection
                - Consider using a VPN if you're in a region with restricted access
                """)
                return None
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            return None
    
    progress_bar.empty()
    status_text.empty()
    return None


# Session state for reading list and settings
if "reading_list" not in st.session_state:
    st.session_state.reading_list = []
if "timeout" not in st.session_state:
    st.session_state.timeout = 15

st.set_page_config(page_title="Profistant AI", page_icon="🎓", layout="wide")
st.title("📚 Profistant - Your Research Kick-Starter")
st.markdown(
    """
    Profistant is an AI-powered research assistant designed to help you kick-start your academic journey. 
    It provides quick access to scholarly articles, summaries, and insights.
    """
)

# Advanced settings sidebar
with st.sidebar:
    st.markdown("### ⚙️ Advanced Settings")
    
    # Proxy configuration
    use_proxy = st.checkbox("Use Proxy (if you're behind a firewall)")
    if use_proxy:
        proxy_host = st.text_input("Proxy Host (e.g., proxy.company.com)")
        proxy_port = st.text_input("Proxy Port (e.g., 8080)")
        
        if proxy_host and proxy_port:
            try:
                scholarly.use_proxy(http=f"http://{proxy_host}:{proxy_port}")
                st.success("✅ Proxy configured!")
            except Exception as e:
                st.error(f"❌ Proxy configuration failed: {str(e)}")
    
    # Timeout settings
    timeout = st.slider("Request Timeout (seconds)", 5, 30, st.session_state.timeout)
    st.session_state.timeout = timeout
    
    st.markdown("---")
    st.markdown("### 📊 App Status")
    st.markdown(f"**Timeout:** {st.session_state.timeout}s")
    st.markdown(f"**Proxy:** {'Enabled' if use_proxy else 'Disabled'}")
    
    st.markdown("---")
    st.markdown("### 📚 Reading List")
    if st.session_state.reading_list:
        st.markdown(f"**Papers saved:** {len(st.session_state.reading_list)}")
        if st.button("🗑️ Clear Reading List"):
            st.session_state.reading_list = []
            st.rerun()
    else:
        st.markdown("No papers saved yet")
#input 
topic = st.text_input("Enter your research topic or area of interest:")

if topic:
    st.write("🔍 Searching for top papers...")
    
    # Use the retry mechanism
    search_results = search_papers_with_retry(topic, timeout=st.session_state.timeout)
    
    if search_results is None:
        st.markdown("---")
        st.markdown("### 🔄 Alternative Options")
        st.markdown("Since we couldn't connect to Google Scholar, you can:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Try Again", key="retry_search"):
                st.rerun()
        
        with col2:
            if st.button("📝 Manual Input", key="manual_input"):
                st.session_state.manual_mode = True
        
        if st.session_state.get("manual_mode", False):
            st.markdown("### 📝 Manual Paper Input")
            st.markdown("Enter paper details manually:")
            
            title = st.text_input("Paper Title:")
            authors = st.text_input("Authors:")
            abstract = st.text_area("Abstract:")
            url = st.text_input("URL (optional):")
            
            if st.button("Add Paper", key="add_manual"):
                if title and authors and abstract:
                    manual_paper = {
                        "title": title,
                        "authors": authors,
                        "abstract": abstract,
                        "url": url if url else "#"
                    }
                    st.session_state.reading_list.append(manual_paper)
                    st.success("✅ Paper added to reading list!")
                    st.session_state.manual_mode = False
                    st.rerun()
                else:
                    st.error("Please fill in at least title, authors, and abstract.")
        
        st.stop()  # Stop execution if search failed
    
    papers = []
    try:
        for i in range(5):  # Top 5 papers
            try:
                paper = next(search_results)
                papers.append({
                    "title": paper.get("bib", {}).get("title", "No title"),
                    "abstract": paper.get("bib", {}).get("abstract", "No abstract available."),
                    "authors": paper.get("bib", {}).get("author", "Unknown"),
                    "url": paper.get("pub_url", "#")
                })
            except StopIteration:
                break
    except Exception as e:
        st.error(f"❌ Error processing search results: {str(e)}")
        st.stop()

    st.write("📝 Found papers:")
    for idx, paper in enumerate(papers):
        st.subheader(f"{idx+1}. {paper['title']}")
        st.markdown(f"**Authors:** {paper['authors']}")
        st.markdown(f"[🔗 View Paper]({paper['url']})")
        st.markdown(f"📄 **Abstract:** {paper['abstract']}")

        if st.button(f"Summarize #{idx+1}", key=f"summarize_{idx}"):
            with st.spinner("Using Gemini to summarize..."):
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents= f"""
                    Summarize the following research abstract into 3 bullet points. Then, suggest one possible research direction or idea a student could explore based on it.
 
                    Abstract:
                    {paper['abstract']}
                    """
                )
                st.markdown("### ✨ Summary + Suggested Research Direction:")
                st.markdown(response.text)
        if st.button(f"Add to Reading List #{idx+1}", key=f"add_{idx}"):
            st.session_state.reading_list.append(paper)
            st.success("✅ Added to reading list!")