# 中文文本具體／抽象二元分類

## 研究問題與重要性

本專案探討句子在語言表達上偏向可感知、可想像的具體場景，還是偏向概念、價值與心理狀態等抽象內容。具體性可能影響讀者如何建構故事中的人物、動作與空間，也可能與敘事的節奏、情緒效果和「故事形狀」有關。以句子為單位建立分類器，可作為探索小說語言差異的初步量化工具，而不是取代細讀或文學詮釋。

訓練資料取自魯迅《呐喊》全部 14 篇小說，應用文本為不同作者郁達夫的〈秋河〉，用以觀察模型跨作者套用時的行為。本次保留原始文字字形，不作簡繁轉換；標註由明確的關鍵詞規則產生，目的是可重現與可稽覈。這只是課堂研究原型，分類結果不代表文學文本具有唯一、客觀的具體／抽象界線。

## 標籤定義與標註規則

`concrete` 指主要描寫可感知的物件、動作、地點、人物或能直接想像的場景；`abstract` 指主要討論概念、價值、關係、心理狀態或評價，缺乏直接可感知的指涉。當同一句同時包含兩類內容時，規則只依線索詞數比較，無法真正判斷句子的主旨。

標註流程分三步，全部可重現：

1. **規則標註**：`scripts/03_ai_label.py` 按抽象與具體線索詞計數對候選句逐句標註；平手時以引號和人稱代詞密度作輔助，仍無法決定者標為 `uncertain`。結果寫入 `data/ai_labels.csv`，同時產生 203 句分層抽查表 `data/manual_check.csv`（每句附規則標籤，供人工填寫 `my_label` 與 `note`）。這不是大型語言模型。
2. **人工複核**：在 `manual_check.csv` 為每句填入 `my_label`（`concrete` 或 `abstract`）。本次 203 句中 168 句與規則一致（一致率 82.76%），35 句分歧已修正。
3. **套用修正並平衡抽樣**：`scripts/03b_balance_sample.py` 先把複核表中的修正套回全量標註，再從兩類各抽 400 句（固定種子 42）組成 800 句平衡樣本 `data/ai_labels_balanced.csv`；若任一類少於 200 句，程式會停止，不合成資料。

檔名與欄名 `03_ai_label.py`、`ai_labels.csv`、`ai_label` 沿用專案早期命名；實際標籤由本機規則產生，沒有呼叫外部 AI 或 LLM。

## 資料來源與著作權說明

來源是專案根目錄提供的 `呐喊.txt` 與 `秋河_(郁達夫).txt`；程式將原始檔複製至 `data/raw/`，之後只讀取複本。訓練池為《呐喊》全部 14 篇切分出的候選句，其他文本不納入模型。魯迅（1881–1936）與郁達夫（1896–1945）的原作在作者逝世超過七十年後，通常屬公有領域；但目前未提供電子檔的版本、整理者或下載網址，因此此說明僅針對原作，不對特定數位版本的來源鏈作額外保證。報告與輸出只引用必要短句，不重刊全文。

## 執行方式

在專案根目錄安裝依賴並依序執行：

```powershell
pip install -r requirements.txt
python scripts\01_split_nahan.py
python scripts\02_prepare_candidates.py
python scripts\03_ai_label.py
# 先在 data\manual_check.csv 完成 my_label 人工複核，再繼續
python scripts\03b_balance_sample.py
python scripts\04_split_data.py
python scripts\05_train_eval.py
python scripts\06_apply_analysis.py
python scripts\07_build_dashboard.py
python scripts\08_build_report_pdf.py
```

`03_ai_label.py` 產生規則標註與人工複核表；先逐句填完 `manual_check.csv` 的 `my_label`（只接受 `concrete` 或 `abstract`），再執行 `03b_balance_sample.py`。若仍有空白或無效標籤，程式會停止，不會默默跳過人工複核。之後會套用修正並平衡抽樣（每類最多 400 句，固定種子 42；若每類少於 200，程式會停止，不合成資料）。`04_split_data.py` 再以 80/20 分層切分。最後的 `07_build_dashboard.py` 將互動頁資料與 CSV 匯出至 `docs/data/`；`08_build_report_pdf.py` 則把 `report.md` 排版成 `report.pdf`（需 `reportlab`）。

本次切句後 2,394 句，初始標註 concrete 1,579、abstract 450、uncertain 365；203 句人工抽查一致率 82.76%，35 個分歧標籤在平衡抽樣前套用。平衡樣本為兩類各 400，共 800，分割為 train 640（各 320）與 test 160（各 80）。篇目分切時〈一件小事〉1,061 字、〈阿Q正傳〉22,155 字；兩篇均保留原文全文，作為自然篇幅例外。字元 TF-IDF n-gram（2–5 字元）在測試集 accuracy 為 0.7562、macro F1 為 0.7560；159/160 句有非零字元特徵。測試標籤仍受規則標註與篇目線索限制。〈秋河〉模型預測 concrete 44 句、abstract 67 句；99/111 句有字元特徵，且 105 句落在 prob_concrete 0.4–0.6，機率接近邊界，須人工解讀而不是視為確定標籤。

## 專案結構

```text
data/raw/       原始文本複本及《呐喊》14 篇切分文本
data/           候選句、初始標註、平衡樣本、人工複核、訓練與分析句資料
scripts/        01–08 處理流程（含規則標註、平衡抽樣與 PDF 產生）
output/         模型指標、測試預測、〈秋河〉預測與 joblib 模型
docs/           GitHub Pages 互動式分析頁及其靜態資料
report.md       研究報告
report.pdf      提交用 PDF（由 scripts/08_build_report_pdf.py 產生）
requirements.txt
```

## 已知限制

規則式標註容易把多義詞或只在特定語境成立的線索當成固定證據，且標籤不等同人工判讀；本次抽查一致率 82.76%，已在平衡抽樣前套用複核標籤。全部訓練篇目仍來自同一作者，而且風格極端的篇目（如偏抽象的〈狂人日記〉、偏具體的〈社戲〉及篇幅特別長的〈阿Q正傳〉）可能令模型利用篇目／作者線索，而非句子自身的抽象性。字元 n-gram 可緩解未斷詞造成的零特徵，但不能消除單一作者、篇目洩漏或規則標註偏誤。具體與抽象連續且可能並存；未來應加入作者分組測試、人工雙人標註、LLM 輔助標註或 BERT 類模型。

## 互動式分析頁

互動頁原始檔為 `docs/index.html`，儀錶板資料由 `python scripts\07_build_dashboard.py` 產生。GitHub Pages 請在 repository Settings → Pages 中選擇部署來源為 `Deploy from a branch`、分支 `main`、資料夾 `/docs`；每次重訓或更改複核表後，重跑腳本 07 並將 `docs/` 更新內容推送，即可更新網站。頁面可比較模型、篩選〈秋河〉預測與測試錯誤、搜尋人工複核分歧，且不使用外部 CDN。
