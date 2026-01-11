# 時系列データのセグメント化の検討

```
時系列データ d0 d1 d2 ... dN があり、状態がきりかわる 
aaaa (中間状態) cccc (中間状態) rrr ... 

以下用件でうまくsegment化したい。

類似度は分かる 別状態 (a vs cや c vs r)なら、類似度は必ず、相当低くなる: thr=0.1以下.
ただし中間状態ではその限りでない. 

また、状態変化のミニマムはだいたいわかる. 
パラメタ N_trans_min 
例: aの中央のindex~cの中央のIndexの差が、最小でN_trans_min くらいはある、という意味

境界は、中間状態あたりにしたい.
できればすべてのペア間の類似度は計算したくない。
```

# 検討結果

前提（使える性質）

* `sim(i,j)=類似度(d_i,d_j)` は計算できる（高いほど同状態っぽい想定）。
* **安定状態同士が別状態なら必ず** `sim ≤ thr(=0.1)`。
* 遷移（中間）ではその限りでない（= 0.1を超えることもある）。
* 別状態の「中心」同士の距離は少なくとも `N_trans_min`。

狙い

* まず「左の安定状態の代表」と「右の安定状態の代表」を少ない比較で見つける
* その2点に対して各時刻が **どちらに近いか** を見て、境界を**中間あたり**（どっちにも決め手がない所）に置く

---

## アルゴリズム：Jump（粗探索） + 2参照点で境界を中央寄せ（精密化）

### パラメータ

* `thr = 0.1`
* `L = floor(N_trans_min/2)`（まずはこれ。迷うなら `L=N_trans_min` でも可）
* `K = max(1, floor(L/4))`（「右代表が遷移を抜けた」っぽい確認用）
* `M = 16`（境界付近の粗いグリッド探索点数）

### 1) 次の状態に入った「右代表」q を、L刻みで飛びながら探す（比較回数を削減）

左代表を `p` とする（初期は `p=0` か、最初の安定そうな点）。

* `q = p + L, p + 2L, p + 3L, ...` と進め、
* 最初に `sim(p,q) ≤ thr` になった `q` を見つける
  これは「p がいる安定状態」と **別の安定状態に到達した可能性が高い**（別安定状態なら必ず低いので）。

ただし `q` が遷移中だとブレるので、追加で：

* `sim(p, q), sim(p, q+1), ... sim(p, q+K-1)` の **最大値**が `≤ thr` になるまで `q` を少し前に進める
  （別安定状態に入っていれば、pとの類似はずっと低いはず、という確認）

→ これで `p`（左安定代表）と `q`（右安定代表）が手に入る。

### 2) 境界を「中間あたり」に置く：g(t)=左っぽさ−右っぽさ のゼロ付近

`p` と `q` が決まったら、区間 `[p, q]` で

* `g(t) = sim(t,p) - sim(t,q)`

を見る。

* 左安定側では `sim(t,p)` が優勢 → `g(t) > 0` になりがち
* 右安定側では `sim(t,q)` が優勢 → `g(t) < 0` になりがち
* 遷移の中央付近は「どっちにも決め手がない」→ `g(t) ≈ 0`

なので、境界は **`|g(t)|` が最小になる t**（または符号が変わる場所）に置くのが自然です。

遷移で単調じゃない場合があるので、頑健にするために：

1. `[p,q]` を `M` 点だけサンプルして `|g|` 最小の `t0` を見つける（粗探索）
2. `t0` の近傍（例えば ±`L/8`）だけを線形走査して最小 `|g|` を取る（精密化）

### 3) 次のセグメントへ

* 境界 `b` を出力
* `p = q` として同じことを繰り返す

---

## 計算量（類似度計算回数）

* 粗探索：だいたい `O(N / L)` 回（L刻みジャンプ）
* 各境界の精密化：`O(M + L/4)` 程度（M点サンプル + 近傍走査。二分探索できるなら `O(log L)` まで落とせる）
* **全ペア `O(N^2)` は不要**

---

## 擬似コード（そのまま実装の骨格になる形）

```python
def segment_by_two_refs(sim, N, thr=0.1, N_trans_min=100):
    L = max(1, N_trans_min // 2)
    K = max(1, L // 4)
    M = 16

    boundaries = []
    p = 0

    def g(t, p, q):
        return sim(t, p) - sim(t, q)

    while True:
        # --- 1) jump search for q ---
        q = p + L
        while q < N and sim(p, q) > thr:
            q += L
        if q >= N:
            break

        # --- 1b) confirm q is beyond transition (stable in new state) ---
        while q + K < N:
            mx = max(sim(p, q+i) for i in range(K))
            if mx <= thr:
                break
            q += 1
        if q >= N:
            break

        # --- 2) boundary refine: minimize |g(t)| ---
        # 2a) coarse grid
        best_t = None
        best_val = float("inf")
        for i in range(1, M):
            t = p + (q - p) * i // M
            val = abs(g(t, p, q))
            if val < best_val:
                best_val = val
                best_t = t

        # 2b) local refine around best_t
        lo = max(p, best_t - L // 8)
        hi = min(q, best_t + L // 8)
        b = lo
        best_val = float("inf")
        for t in range(lo, hi + 1):
            val = abs(g(t, p, q))
            if val < best_val:
                best_val = val
                b = t

        boundaries.append(b)

        # next segment
        p = q

    return boundaries
```

---

## 使い分けの目安

* `L=N_trans_min/2`：境界を取りこぼしにくい（比較回数はやや増える）
* `L=N_trans_min`：比較回数を減らせる（ただし遷移が広い/セグメントが短いと代表点が取りづらい場合あり）
* 遷移が長くて `q` がなかなか「安定」にならないなら、`K` を少し大きくする（例：`K=L/2`）

---

この方式の要点は、「別安定状態なら必ず `≤thr`」を **代表点（p,q）発見の確証**に使い、境界は `sim(t,p)` と `sim(t,q)` の **拮抗点（|g|最小）**として置くので、要求どおり「中間状態あたり」に寄せられることです。
