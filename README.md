# デスクトップアプリ提案書（スマホ動画 → ページ画像抽出 → 超速読用要約の前処理）

## 1. 目的

スマホで撮影した「本をパラパラめくる動画」から、ページごとの代表フレーム（画像）を自動抽出し、後段（Vision/OCR/LLM）で要点要約できる入力を生成するデスクトップアプリを提供する。

* 完璧なスキャン品質は不要（超速読・流し読み用）
* 重複や多少の取りこぼしは許容
* 学習（自前モデル訓練）は行わない

---

## 2. 想定ユーザー操作（MVP）

1. 動画（mp4/mov）をドラッグ＆ドロップ
2. 「ページ抽出」を実行
3. 抽出されたページサムネイル一覧を確認
4. 必要なら手動で削除/追加/順序入替
5. 画像フォルダ（＋任意でPDF）にエクスポート
   （後段の要約処理は別モジュール/別工程として接続可能にする）

---

## 3. スコープ

### In Scope

* 動画からページ画像抽出（本提案の中核）
* 透視補正（可能なら）＋フォールバック（補正失敗でも出力）
* UIでの結果確認・簡易編集
* エクスポート（画像/JSONメタデータ）

### Out of Scope（MVPではやらない）

* 湾曲補正（見開き曲面の完全補正）
* 手や指の完全除去（学習なしだと難易度が上がる）
* OCR/要約の品質最適化（後段で改善）
* リアルタイム（ライブカメラ）処理（まずはオフライン動画処理）

---

## 4. アーキテクチャ概要

### コンポーネント

* **UI層**：動画選択、進捗表示、サムネイルグリッド、簡易編集、エクスポート
* **処理パイプライン層**：

  * 動画デコード（FFmpeg/OS標準デコーダ）
  * 低解像度・低fps解析（ページ境界検出）
  * 候補フレーム選択（ページ代表）
  * （任意）ページ四角検出→透視補正
  * 出力生成（画像＋メタデータ）
* **設定層**：パラメータ（fps、閾値、探索窓等）をプリセット化

### 実装技術候補（実装者裁量）

* 言語：Python + OpenCV（最短） / C++ + OpenCV（速度） / Node/Electron + ネイティブ処理（GUI強）
* GUI：PySide6 / Qt / Electron
* 動画入出力：OpenCV VideoCapture（簡便だがコーデック相性注意） or FFmpeg（堅い）

---

## 5. 中核アルゴリズム：静止しない“パラパラ動画”からページ代表フレーム抽出

静止区間がほぼ無い前提のため、方針は以下。

* **ページめくりは「動き量のピーク（PCE）」として出る**
* 各ピーク直後（新しいページが見え始める）短い探索窓から、**最もマシなフレームを1枚**選ぶ
* 完璧不要なので、補正失敗時もフォールバックで画像出力する

### 5.1 解析用フレーム列の生成

* 入力動画：高fps推奨（可能なら60fps）
* 解析：`10–15 fps` に間引き、長辺 `~640px` に縮小
  → 高速・頑健（露出/ノイズの影響を平均化しやすい）

### 5.2 ページめくりイベント（PCE）検出：差分の“モーション・ブロブ”時系列

各解析フレーム `F_t` について以下を計算：

1. `D_t = abs(F_t - F_{t-1})`（グレースケール推奨）
2. 平滑化（Gaussian）
3. 小ノイズ抑制（erode / open）
4. 二値化（固定閾値 or 自動閾値）
5. `A_t = countNonZero(D_t_bin)`（動き量スカラー）
6. **OR積算**：`M_t = OR(D_{t}, D_{t-1}, ..., D_{t-(N-1)})`（N=6程度）
   `B_t = countNonZero(M_t)`（“動きの塊”の面積）

PCE候補は `B_t` の局所最大（ピーク）。
ノイズ対策として：

* `B_t` に移動平均/メディアンで平滑化
* ピーク間の最小間隔（`min_peak_distance`）を設定（例：0.1–0.2秒相当）
* ピーク高さの閾値は固定値より「分布に対する相対値（例：中央値＋k*MAD）」で自動化

### 5.3 ピーク後の探索窓で代表フレームを選ぶ（静止不要）

各ピーク `p_k` ごとに、探索窓 `[p_k + s0, p_k + s1]` を定義し、その範囲から1枚選ぶ。

* 推奨：`s0=1 frame`（直後すぎる崩れ回避）、`s1=~0.4秒`（めくり速度により調整）
* 選ぶ基準（学習なしスコア）：

  * `sharpness(t)`：Laplacian variance（大きいほど良い）
  * `residual_motion(t)`：`B_t` または `A_t`（小さいほど良い）
  * `quad_ok(t)`：ページ四角検出成功（成功=1/失敗=0、または面積スコア）

例：
`score(t) = sharpness(t) - a*residual_motion(t) + b*quad_ok(t)`

出力は基本1枚。取りこぼしが気になる場合のみ、上位2枚を出して後段に渡す（重複許容なので安全）。

### 5.4 ページ四角検出→透視補正（任意・失敗許容）

候補フレーム（代表）に対してのみ高解像度フレームを取得し、以下を試す：

* エッジ抽出 → 輪郭 → 最大四角形（近似） → 4点整列 → perspective warp
* 成功条件（例）：

  * 四角面積が画面の一定割合以上
  * 凸四角形
  * 長方形として極端に潰れていない

**フォールバック（必須）**

* 四角が取れない場合：元フレームをそのまま保存
* 追加フォールバック（任意）：中央大きめクロップを保存（文字領域が中央に寄るケースで有効）

---

## 6. 出力仕様

* `output/pages/page_0001.png` …（抽出画像）
* `output/pages/page_0001.json` …（メタデータ）

  * `source_video_timestamp_ms`
  * `pce_peak_index`
  * `score_components`（sharpness/motion/quad）
  * `warp_applied`（true/false）
  * `quad_points`（あれば）
* （任意）`output/pages.pdf`（後段がPDFを受けやすい場合）

---

## 7. UI仕様（MVP）

* 左：入力動画・設定（プリセット：Fast / Balanced / Robust）
* 中：進捗（PCE検出→抽出→補正→出力）
* 右：サムネイル一覧（ページ番号・タイムスタンプ表示）

  * 操作：削除、複製、順序入替、手動追加（任意でフレーム指定）

---

## 8. パラメータ（初期値案）

* 解析fps：10–15
* 解析解像度：長辺 640
* OR積算N：6
* ピーク最小間隔：0.12秒相当
* 探索窓：`[+1 frame, +0.4秒]`
* 出力枚数：1（不安なら2）
* フォールバック：ON（必須）

---

## 9. 品質基準（受け入れ条件）

* “静止しないパラパラ動画”から、ページらしい画像が一定数出力される
  * 例：100ページ相当のめくりで、50–150枚程度の抽出（重複・欠落許容）
* 処理がクラッシュせず完走し、サムネイル確認とエクスポートができる
* 四角補正が失敗しても必ず画像は出る（フォールバック）

---

## 10. リスクと対策

* **常時大きく揺れる/ブレる**：ピークが埋もれる
  → `B_t` の自動閾値化（中央値＋MAD）、pHash差分による補助境界（任意）
* **照明反射・白飛び**：輪郭検出が不安定
  → 透視補正は“できたらやる”、失敗はフォールバックで通す
* **手の遮蔽**：補正やOCRの邪魔
  → MVPは許容。後段要約で吸収／将来は軽量手検出（学習済みモデル利用）を検討

---

## 11. 実装者向け疑似コード（中核のみ）

```python
# 解析フレーム列の生成（縮小・間引き）
frames = decode_video(video_path, fps=12, long_side=640, gray=True)

# motion blob time series
B = []
bin_diffs = []
for t in range(1, len(frames)):
    D = absdiff(frames[t], frames[t-1])
    D = gaussian(D)
    D = erode_or_open(D)
    Dbin = threshold(D)
    bin_diffs.append(Dbin)

# OR accumulation (N=6)
N = 6
for t in range(len(bin_diffs)):
    M = OR(bin_diffs[max(0, t-N+1):t+1])
    B.append(count_nonzero(M))

# smooth and peak-pick
B_s = smooth(B)
peaks = pick_peaks(B_s, min_distance=0.12s, height=auto_threshold(B_s))

# for each peak, choose best frame in post-window
selected = []
for p in peaks:
    window = range(p+1, p + int(0.4s * fps))
    best = argmax(window, lambda i: score(frames[i], B_s[i], quad_ok(frames[i])))
    selected.append(best)

# for each selected index, extract high-res frame and optionally warp
for idx in selected:
    hi = decode_frame_highres(video_path, timestamp=ts(idx))
    quad = detect_page_quad(hi)
    if quad.success:
        out = warp_perspective(hi, quad)
    else:
        out = fallback_crop_or_original(hi)
    save(out)
```

---

## 12. 将来拡張（必要になったら）

* 代表フレーム選択を「2枚出力→後段LLMでベスト選択」に寄せる（超速読用途に強い）
* 見開き自動分割
