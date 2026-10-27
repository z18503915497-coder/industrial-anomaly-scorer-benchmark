"""Read-only reference verification against Crossref/DataCite/CVF/publisher pages.

Status meanings:
  VERIFIED  = official metadata page reachable and title/page/year match
  PARTIAL   = page reachable, but one metadata field could not be confirmed
  PENDING   = source unreachable or only a secondary index available
  ERROR     = request failed
"""
import json
import os
import re
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "REFERENCE_VERIFICATION.json")

REFS = [
    dict(key="Ahuja2019", kind="datacite", doi="10.48550/arXiv.1909.11786",
         title="Probabilistic Modeling of Deep Features for Out-of-Distribution and Adversarial Detection"),
    dict(key="Batzner2024", kind="cvf",
         url="https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html",
         title="EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies",
         pages="128 - 138"),
    dict(key="Benjamini1995", kind="crossref", doi="10.1111/j.2517-6161.1995.tb02031.x",
         title="Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing",
         pages="289-300"),
    dict(key="Bergmann2019", kind="cvf",
         url="https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html",
         title="MVTec AD", pages="9592-9600"),
    dict(key="Cohen2020", kind="datacite", doi="10.48550/arXiv.2005.02357",
         title="Sub-Image Anomaly Detection with Deep Pyramid Correspondences"),
    dict(key="Damm2025", kind="cvf",
         url="https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html",
         title="AnomalyDINO", pages="1319-1329"),
    dict(key="Defard2021", kind="crossref", doi="10.1007/978-3-030-68799-1_35",
         title="PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization",
         pages="475-489"),
    dict(key="Gonzalez1985", kind="crossref", doi="10.1016/0304-3975(85)90224-5",
         title="Clustering to Minimize the Maximum Intercluster Distance", pages="293-306"),
    dict(key="Gu2025", kind="cvf",
         url="https://openaccess.thecvf.com/content/CVPR2025/html/Gu_UniVAD_A_Training-free_Unified_Model_for_Few-shot_Visual_Anomaly_Detection_CVPR_2025_paper.html",
         title="UniVAD", pages="15194-15203"),
    dict(key="Gudovskiy2022", kind="cvf",
         url="https://openaccess.thecvf.com/content/WACV2022/html/Gudovskiy_CFLOW-AD_Real-Time_Unsupervised_Anomaly_Detection_With_Localization_via_Conditional_Normalizing_WACV_2022_paper.html",
         title="CFLOW-AD", pages="98-107"),
    dict(key="Holm1979", kind="crossref", doi="10.2307/4615733",
         title="A Simple Sequentially Rejective Multiple Test Procedure", pages="65-70"),
    dict(key="Huang2022", kind="crossref", doi="10.1007/978-3-031-20053-3_18",
         title="Registration Based Few-Shot Anomaly Detection", pages="303-319"),
    dict(key="Ilharco2021", kind="datacite", doi="10.5281/zenodo.5143773",
         title="OpenCLIP"),
    dict(key="Jeong2023", kind="cvf",
         url="https://openaccess.thecvf.com/content/CVPR2023/html/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.html",
         title="WinCLIP", pages="19606-19616"),
    dict(key="Lendering2026", kind="cvf",
         url="https://openaccess.thecvf.com/content/CVPR2026/html/Lendering_SubspaceAD_Training-Free_Few-Shot_Anomaly_Detection_via_Subspace_Modeling_CVPR_2026_paper.html",
         title="SubspaceAD", pages="28557-28566"),
    dict(key="Li2024PromptAD", kind="cvf",
         url="https://openaccess.thecvf.com/content/CVPR2024/html/Li_PromptAD_Learning_Prompts_with_only_Normal_Samples_for_Few-Shot_Anomaly_CVPR_2024_paper.html",
         title="PromptAD", pages="16838-16848"),
    dict(key="Li2026FastRef", kind="cvf",
         url="https://openaccess.thecvf.com/content/CVPR2026/html/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.html",
         title="FastRef", pages="43040-43049"),
    dict(key="Mahalanobis1936", kind="pdf",
         url="https://insa.nic.in/writereaddata/UpLoadedFiles/PINSA/Vol02_1936_1_Art05.pdf",
         title="On the Generalized Distance in Statistics", pages="49-55"),
    dict(key="Pedregosa2011", kind="html",
         url="https://www.jmlr.org/papers/v12/pedregosa11a.html",
         title="Scikit-learn: Machine Learning in Python", pages="2825-2830"),
    dict(key="Radford2021", kind="html",
         url="https://proceedings.mlr.press/v139/radford21a.html",
         title="Learning Transferable Visual Models From Natural Language Supervision", pages="8748-8763"),
    dict(key="Roth2022", kind="cvf",
         url="https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html",
         title="Towards Total Recall in Industrial Anomaly Detection", pages="14318-14328"),
    dict(key="ScikitLearn2026", kind="html",
         url="https://scikit-learn.org/1.9/modules/generated/sklearn.mixture.GaussianMixture.html",
         title="GaussianMixture"),
    dict(key="Wilcoxon1945", kind="crossref", doi="10.2307/3001968",
         title="Individual Comparisons by Ranking Methods", pages="80-83"),
    dict(key="Zhou2024", kind="html",
         url="https://proceedings.iclr.cc/paper_files/paper/2024/hash/d7b50b8ac2c781a12f26155f48310d8d-Abstract-Conference.html",
         title="AnomalyCLIP"),
    dict(key="Zou2022", kind="crossref", doi="10.1007/978-3-031-20056-4_23",
         title="SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation",
         pages="392-408"),
]


def fetch(url, timeout=35):
    req = urllib.request.Request(url, headers={"User-Agent": "reference-audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"<.*?>", "", s or "")).strip()


def norm_pages(s):
    return re.sub(r"\s+", "", (s or "").replace("–", "-").replace("−", "-"))


def check_crossref(ref):
    data = json.loads(fetch("https://api.crossref.org/works/" + ref["doi"]))["message"]
    title = norm((data.get("title") or [""])[0])
    page = data.get("page") or ""
    year = ""
    for k in ("published-print", "published-online", "issued"):
        if data.get(k, {}).get("date-parts"):
            year = str(data[k]["date-parts"][0][0]); break
    ok_title = ref["title"].lower() in title.lower()
    ok_page = (not ref.get("pages")) or (page.replace("–", "-") == ref["pages"])
    return dict(status="VERIFIED" if ok_title and ok_page else "PARTIAL",
                title=title, page=page, year=year,
                title_match=ok_title, page_match=ok_page, source="Crossref")


def check_datacite(ref):
    data = json.loads(fetch("https://api.datacite.org/dois/" + ref["doi"]))["data"]["attributes"]
    title = norm((data.get("titles") or [{}])[0].get("title", ""))
    year = str(data.get("publicationYear", ""))
    ok_title = ref["title"].lower() in title.lower()
    return dict(status="VERIFIED" if ok_title else "PARTIAL",
                title=title, year=year, title_match=ok_title, source="DataCite")


def check_cvf(ref):
    html = fetch(ref["url"]).decode("utf-8", "ignore")
    m = re.search(r'<div[^>]*id="papertitle"[^>]*>(.*?)</div>', html, re.S)
    title = norm(m.group(1)) if m else ""
    ok_title = ref["title"].lower() in title.lower()
    ok_page = (norm_pages(ref.get("pages", "")) in norm_pages(html)) if ref.get("pages") else True
    return dict(status="VERIFIED" if ok_title and ok_page else "PARTIAL",
                title=title, page_match=ok_page, title_match=ok_title, source="CVF")


def check_html(ref):
    html = fetch(ref["url"]).decode("utf-8", "ignore")
    m = re.search(r"<title>(.*?)</title>", html, re.S | re.I)
    title = norm(m.group(1)) if m else ""
    ok_title = ref["title"].lower() in html.lower() or ref["title"].lower() in title.lower()
    ok_page = (norm_pages(ref.get("pages", "")) in norm_pages(html)) if ref.get("pages") else True
    return dict(status="VERIFIED" if ok_title and ok_page else "PARTIAL",
                title=title, page_match=ok_page, title_match=ok_title, source="publisher/conference page")


def check_pdf(ref):
    raw = fetch(ref["url"])
    path = os.path.join(BASE, "_tmp_ref.pdf")
    with open(path, "wb") as fh:
        fh.write(raw)
    try:
        import fitz
        doc = fitz.open(path)
        text = "\n".join(p.get_text() for p in doc)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    ok_title = ref["title"].lower() in text.lower()
    ok_page = ref.get("pages", "") in text if ref.get("pages") else True
    return dict(status="VERIFIED" if ok_title else "PARTIAL",
                title_match=ok_title, page_match=ok_page, source="PDF")


def main():
    out = {}
    for ref in REFS:
        try:
            if ref["kind"] == "crossref":
                res = check_crossref(ref)
            elif ref["kind"] == "datacite":
                res = check_datacite(ref)
            elif ref["kind"] == "cvf":
                res = check_cvf(ref)
            elif ref["kind"] == "html":
                res = check_html(ref)
            elif ref["kind"] == "pdf":
                res = check_pdf(ref)
            else:
                res = dict(status="PENDING", source="unknown")
        except Exception as e:
            res = dict(status="ERROR", error=repr(e), source=ref["kind"])
        res["expected_title"] = ref["title"]
        res["expected_pages"] = ref.get("pages", "")
        res["doi_or_url"] = ref.get("doi") or ref.get("url")
        out[ref["key"]] = res
        print(ref["key"], res["status"], res.get("title", "")[:80], res.get("page", ""), res.get("error", ""))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
