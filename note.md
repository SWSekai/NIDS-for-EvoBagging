### Working
* 融合EvoBagging方法
### Future work
* TomekLinks(maybe)
* 了解EvoBagging代入方法(程式) -> now
* 確認實驗比例(參數) -> now
* 確認使用模型
### 參數設定邏輯
\* 依賴意即可以不傳入
  1. **n_select(選擇的袋子數量)**:建議設定為 n_new_bags 的 1 到 2 倍，待實驗(依賴)
  2. **n_new_bags(新生成袋子數量)**:根據論文，以5作為預設值
  3. **max_initial_size (初始袋子最大尺寸)**:影響訓練成本和性能，論文提出的100%無法在大型資料集上有效呈現，將從10%開始向上確認
  4. **n_crossover (交配的袋子數量)**: 建議設定在 n_select 的一半左右
  5. **n_mutation (突變的袋子數量)**:
### 缺失
* 捨棄正常流量的前提是二階段NIDS
### 思路
* 考慮剔除mutation機制，原因：從論文中並未見到其明顯效果，待實驗證實