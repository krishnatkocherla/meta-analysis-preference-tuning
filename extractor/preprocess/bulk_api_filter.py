import os
import logging
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from dotenv import load_dotenv
import os
import hashlib

def _stable_dest(url: str, download_dir: str) -> str:
    """
    Create a deterministic filename for a given URL, ensuring uniqueness.
    """
    filename = os.path.basename(url)
    if not filename.endswith(".pdf"):
        # Use hash to prevent collisions if filename is not clean
        h = hashlib.md5(url.encode()).hexdigest()[:10]
        filename = f"{h}.pdf"
    return os.path.join(download_dir, filename)

def _download_to_path(url: str, download_dir: str, session: Session) -> str:
    """
    Download a PDF from the given URL and save it in `download_dir`.
    Returns the path to the downloaded file.
    """
    dest = _stable_dest(url, download_dir)
    if os.path.exists(dest):
        logging.info(f"Already exists: {dest}")
        return dest

    headers = {"User-Agent": "SemanticScholarDownloader/1.0"}

    try:
        with session.get(url, stream=True, timeout=30, headers=headers) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 14):
                    if chunk:
                        f.write(chunk)
        logging.info(f"Downloaded: {dest}")
        return dest
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        return None

# https://github.com/allenai/s2-folks/blob/main/examples/Webinar%20Code%20Examples/API_Bulk_Search.py

# Fetch the API key from environment variables
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(env_path)

API_KEY = os.getenv('API_KEY')

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="[%(name)s] %(message)s",
    datefmt="[%X]",
)

if __name__ == "__main__":
    if not API_KEY:
        raise ValueError("API key not found in environment variables.")

    http = Session()
    http.mount('https://', HTTPAdapter(max_retries=Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"HEAD", "GET", "OPTIONS"}
    )))
    # Expanded keyword list for preference tuning in LLMs
    KEYWORDS = [
        # Direct/Implicit Preference Optimization variants
        "PPO", "Proximal Preference Optimization",
        "DPO", "Direct Preference Optimization",
        "IPO", "Implicit Preference Optimization",
        "f-DPO", "f-divergence Direct Preference Optimization",
        "KTO", "Kahneman-Tversky Optimization",
        "ORPO", "Odds Ratio Preference Optimization",
        "SimPO", "Simple Preference Optimization",
        "RRHF", "Rank Responses with Human Feedback",
        "CPL", "Contrastive Preference Learning",
        "RPO", "Relative Preference Optimization",
        "OPPO", "Offline Preference-guided Policy Optimization",
        "IRPO", "Iterative Reasoning Preference Optimization",
        "SLiC", "SLiC-HF", "Sequence Likelihood Calibration with Human Feedback",
        "GPO", "Generalized Preference Optimization",
        "GRPO", "Group Relative Policy Optimization",
        "Preference-aware fine-tuning", "preference-conditioned fine-tuning",

        # General preference optimization terms
        "preference optimization", "preference tuning", "preference-based fine-tuning",
        "preference-based RL", "preference learning", "preference-aware training",
        "ranking-based fine-tuning", "preference signal", "preference supervision",
        "reward shaping", "policy shaping", "feedback shaping", "guided fine-tuning"
    ]

    def quote_if_phrase(keyword):
        return f'"{keyword}"' if ' ' in keyword else keyword

    query = ' | '.join([quote_if_phrase(k) for k in KEYWORDS])

    # Define the search query and filters
    fields = "paperId,externalIds,url,title,year,openAccessPdf"
    fields_of_study = "Computer Science"  # Filter by fields of study

    token = None
    pdf_urls = []

    while True:
        response = http.get(
            "https://api.semanticscholar.org/graph/v1/paper/search/bulk",
            headers={'x-api-key': API_KEY},
            params={
                'query': query,
                'token': token,
                'fields': fields,
                'fieldsOfStudy': fields_of_study,
            }
        )

        response.raise_for_status()
        data = response.json()
        papers = data.get("data", [])

        for p in papers:
            eid = p.get("externalIds", {})
            if eid == {}: continue

            if 'ArXiv' in eid:
                pdf_url = f"https://arxiv.org/pdf/{eid['ArXiv']}.pdf"
                pdf_urls.append(pdf_url)
                logging.info(f"Paper ID: {p['paperId']}, Title: {p['title']}, PDF URL: {pdf_url}")
                continue

            if 'ACL' in eid:
                pdf_url = f"https://aclanthology.org/{eid['ACL']}.pdf"
                pdf_urls.append(pdf_url)
                logging.info(f"Paper ID: {p['paperId']}, Title: {p['title']}, PDF URL: {pdf_url}")
                continue

        logging.info(f"Total estimated matches: {data.get('total')}")
        

        token = data.get('token')
        if not token:
            break
        logging.info(f"Continuation Token: {token}")
    logging.info(f"Total PDF URLs collected: {len(pdf_urls)}")

    for pdf_url in pdf_urls:
        download_dir = "../../downloaded_pdfs"
        os.makedirs(download_dir, exist_ok=True)
        path = _download_to_path(pdf_url, download_dir, http)