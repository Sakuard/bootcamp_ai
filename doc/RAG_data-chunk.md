## RAG Optimization: Data-Chunk

檢索增強生成
文章難易度：★★★☆☆

:star: 本篇著重在
- **實作 RAG 遇到的問題、對應及優化**

:star: 過程中會提供實際範例

## :question: 細說 Embedding
:point_right: Embedding 是將一個物件(ex: 單詞、句子、整篇文章、一個檔案...)向量化的過程(結果通常為一個矩陣)

:point_right: 過程通常涉及
- *Tokenize*：分詞
如：I have a book 是一個 4 個 token 的句子分別是 ['I', 'have', 'a', 'book']
    
- *Vectorize*：向量化
將經過 tokenize 之後的 Ojbect，向量化轉置為一個 向量矩陣
:point_right: Embedding 的目的是為了「向量比對」


## 繁體中文的分詞

:point_right: 通常 Tokenize 會有相關的 model (tokenizer)
但，有些 tokenizer 在處理「中文」的時候，卻會出現下列的情況

    例如： 我有一顆蘋果
    
    得到的結果可能是
    ['我', '有', '一', '顆', '蘋', '果']
    
    但我們預期的是
    ['我', '有', '一', '顆', '蘋果']

:exclamation: 這會直接影響 *向量比對* 的結果

#### 結論
:heavy_check_mark: 使用支援 multi-lingual 的 Embedding-model，如 llama3

## Lost-in-the-middle

目前 LLM 對於過長的內容，會相對關注「開頭」與「結尾」的內容，就會出現 "Lost in the middle" 的情況

為此，我們會對這些資料做 *chunk*

#### Data Chunk
:point_right: 資料分割，控制每次 Embedding 的 context length
:heavy_check_mark: 藉此提高文意捕捉的精準度

:hammer_and_wrench: 初略的實踐方式為
1. 設置基礎的 字元輸入單位
2. 每次 位移 幾個字元單位

```python=
long_content = "這邊是一個長文本內容的範例"

base_context_length = 4
shift_length = 2

# 迴圈切割
for start in range(0, len(long_content) - base_context_length + 1, shift_length):
    chunk = long_content[start:start + base_context_length]
    
    print(chunk)
    print("-" * 20)

```

輸出：

```bash=
這邊是一
--------------------
是一個長
--------------------
個長文本
--------------------
文本內容
--------------------
內容的範
--------------------
```
#### 但還是會遇到下列情況
:page_with_curl: Dataset
```
[
'這是一個關於在哥倫比亞讀 PHD 的經驗分享',
'這是關於在中國遊學的經驗分享'
]
```
:speech_balloon: 我想知道美洲遊學
:point_down: 
相似度比對結果可能是
```
「中國遊學」>「哥倫比亞讀 PHD」
```

:heavy_check_mark: 因此 Data Chunk 的控制，也會直接影響到相似度比對的效率

---
### Reference
- [使用繁體中文評測個家 Embedding 模型的檢索能力](https://ihower.tw/blog/archives/12167)

關鍵字 : AI, ML, NLP, RAG, Embeddings, Data Chunk