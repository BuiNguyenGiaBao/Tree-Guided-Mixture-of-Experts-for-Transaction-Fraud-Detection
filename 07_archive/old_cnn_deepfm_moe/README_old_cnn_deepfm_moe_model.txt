README – Old CNN + DeepFM + MoE Model Files

Gói này là bộ file của model cũ trước bản final proposed/tuned.

Hướng model cũ:
CNN branch + DeepFM branch + Gated MoE fusion

Các file chính:
1. cnn_branch_updated.py
   - CNN branch / CNNMix branch để trích xuất đặc trưng numerical/sequence-like.

2. deepfm_branch_updated.py
   - DeepFM branch để học feature interaction.

3. moe_gated_fusion.py
   - Gated Mixture-of-Experts fusion giữa CNN branch và DeepFM branch.

4. fraud_losses.py
   - Các loss dùng cho fraud detection như BCE/Focal/SupCon tùy runner.

5. train_cnn_deepfm_moe_ablation.py
   - Runner cũ cho ablation:
     cnn_only_bce
     deepfm_only_bce
     concat_bce
     moe_bce
     moe_focal
     moe_focal_supcon

6. train_cnn_deepfm_moe_ablation.ipynb
   - Notebook tương ứng của ablation runner.

7. unified_cnn_deepfm_moe_train.py
   - Runner unified cũ cho CNN + DeepFM + MoE.

8. unified_cnn_deepfm_moe_train.ipynb
   - Notebook tương ứng của unified runner.

9. dl_moe_results_final.csv
   - Kết quả cũ nếu cần đối chiếu với model mới.

Cách dùng:
- Đặt tất cả file .py cùng một thư mục.
- Dữ liệu vẫn dùng output đã xử lý trước đó trong:
  D:\project\data\merge_paper_ready_tree_cost
  hoặc folder mà bạn đã cấu hình trong script.

Lưu ý:
- Gói này KHÔNG phải bản final tree-guided KD + Focal + SupCon tuned mới.
- Model cũ nên đưa vào paper như:
  base proposed model / ablation / intermediate variant.
- Model mới nên là final proposed model.
