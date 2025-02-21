# 毕业设计
毕设存档

pip3 install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu101
启动：
conda activate bysj
D:
cd D:\learn\bysj
cd D:\learn\bysj\webtool
python app.py

图片：
1024*1024
{class_id} {x_center} {y_center} {width} {height}

类别：
{0: 'title', 1: 'plain text', 2: 'abandon', 3: 'figure', 4: 'figure_caption', 5: 'table', 6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption'}

 更新至github：
 git add .
 git commit -m ""
 git push


训练
python train.py --data  --model  --epoch 50 --image-size 416 --batch-size 16 --project 


 