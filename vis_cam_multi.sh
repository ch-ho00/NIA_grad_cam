python3 visualize_cam.py --cfg '/content/drive/MyDrive/MPhil/NIA_insta/configs/multiCeleb_noRef.yml' \
                         --checkpoint '/content/drive/MyDrive/MPhil/NIA_insta/output/multiCeleb_noRef/output/Insta_Trend_Dataset/custom/best_model_state.pth.tar' \
                         --threshold 0.2 \
                         --savedir '/content/drive/MyDrive/MPhil/pytorch_grad_cam/result/noRef' \
                         --prefix 'multi'
                          
# python3 visualize_cam.py --cfg '/content/drive/MyDrive/MPhil/NIA_insta/configs/multiCeleb_withRef.yml' \
#                          --checkpoint '/content/drive/MyDrive/MPhil/NIA_insta/output/multiCeleb_withRef/output/Insta_Trend_Dataset/custom/best_model_state.pth.tar' \
#                          --threshold 0.2 \
#                          --savedir '/content/drive/MyDrive/MPhil/pytorch_grad_cam/result/withRef' \
#                          --prefix 'multi'
