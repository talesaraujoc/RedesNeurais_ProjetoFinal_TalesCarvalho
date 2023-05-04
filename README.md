# RedesNeurais_ProjetoFinal_TalesCarvalho

### Tales Araújo Carvalho

|**Detecção de Objetos**|**YOLOv5|PyTorch**|

## Performance

O modelo treinado possui performance de **98.8%**.

### Output do bloco de treinamento

<details>
  <summary>Click to expand!</summary>
  
  ```text
   train: weights=yolov5s.pt, cfg=, data=custom_data.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=100, batch_size=16, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
github: up to date with https://github.com/ultralytics/yolov5 ✅
requirements: /content/requirements.txt not found, check failed.
YOLOv5 🚀 v7.0-162-gc3e4e94 Python-3.10.11 torch-2.0.0+cu118 CUDA:0 (Tesla T4, 15102MiB)

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
ClearML: run 'pip install clearml' to automatically track, visualize and remotely train YOLOv5 🚀 in ClearML
Comet: run 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=4

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1     24273  models.yolo.Detect                      [4, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model summary: 214 layers, 7030417 parameters, 7030417 gradients, 16.0 GFLOPs

Transferred 343/349 items from yolov5s.pt
AMP: checks passed ✅
optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
train: Scanning /content/dataset/train/labels.cache... 210 images, 0 backgrounds, 0 corrupt: 100% 210/210 [00:00<?, ?it/s]
train: Caching images (0.2GB ram): 100% 210/210 [00:01<00:00, 140.19it/s]
val: Scanning /content/dataset/valid/labels.cache... 60 images, 0 backgrounds, 0 corrupt: 100% 60/60 [00:00<?, ?it/s]
val: Caching images (0.1GB ram): 100% 60/60 [00:01<00:00, 49.65it/s]

AutoAnchor: 3.29 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Plotting labels to runs/train/exp2/labels.jpg... 
Image sizes 640 train, 640 val
Using 2 dataloader workers
Logging results to runs/train/exp2
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       0/99       3.5G     0.1159    0.02889    0.04875          7        640: 100% 14/14 [00:06<00:00,  2.11it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.53it/s]
                   all         60         61    0.00285      0.697     0.0327    0.00661

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       1/99      3.94G    0.08697    0.02846    0.04381          2        640: 100% 14/14 [00:03<00:00,  4.45it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.99it/s]
                   all         60         61    0.00362      0.981     0.0927     0.0216

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       2/99      3.94G    0.06983    0.02697    0.03928          2        640: 100% 14/14 [00:02<00:00,  4.91it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.58it/s]
                   all         60         61     0.0112       0.22     0.0342     0.0104

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       3/99      3.95G    0.06765    0.02382    0.04003          5        640: 100% 14/14 [00:02<00:00,  4.88it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.60it/s]
                   all         60         61     0.0403       0.57     0.0811     0.0281

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       4/99      3.95G    0.06369    0.02088    0.03653          3        640: 100% 14/14 [00:02<00:00,  4.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.86it/s]
                   all         60         61     0.0686      0.581      0.117     0.0315

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       5/99      3.95G    0.06161    0.02164    0.03761          7        640: 100% 14/14 [00:03<00:00,  4.65it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.55it/s]
                   all         60         61      0.182      0.493      0.217     0.0676

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       6/99      3.95G    0.05869    0.02027    0.03553          6        640: 100% 14/14 [00:02<00:00,  5.02it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  4.13it/s]
                   all         60         61        0.2      0.597      0.364      0.115

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       7/99      3.95G    0.05531     0.0191    0.03297          3        640: 100% 14/14 [00:02<00:00,  4.80it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.45it/s]
                   all         60         61      0.645      0.585      0.666      0.236

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       8/99      3.95G    0.05457    0.01667    0.03114          5        640: 100% 14/14 [00:03<00:00,  4.08it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.32it/s]
                   all         60         61      0.771      0.633      0.719       0.27

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       9/99      3.95G    0.05053    0.01627    0.03185          3        640: 100% 14/14 [00:02<00:00,  4.98it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.63it/s]
                   all         60         61      0.478      0.705      0.693      0.253

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      10/99      3.95G    0.04776    0.01477    0.03019          3        640: 100% 14/14 [00:02<00:00,  4.89it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.96it/s]
                   all         60         61      0.639      0.693      0.661      0.337

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      11/99      3.95G    0.04496    0.01401    0.02466          6        640: 100% 14/14 [00:03<00:00,  3.75it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.40it/s]
                   all         60         61       0.44       0.92      0.663      0.282

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      12/99      3.95G    0.04374    0.01259    0.02391          3        640: 100% 14/14 [00:02<00:00,  4.98it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.88it/s]
                   all         60         61      0.593      0.869      0.834      0.413

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      13/99      3.95G    0.04808    0.01383    0.02343          3        640: 100% 14/14 [00:03<00:00,  4.56it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.79it/s]
                   all         60         61      0.672      0.923      0.736      0.444

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      14/99      3.95G    0.04482    0.01438    0.02219          4        640: 100% 14/14 [00:03<00:00,  4.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.95it/s]
                   all         60         61      0.811       0.87      0.934       0.42

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      15/99      3.95G    0.04233    0.01402    0.02155          4        640: 100% 14/14 [00:02<00:00,  5.00it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.63it/s]
                   all         60         61      0.815      0.942      0.867      0.519

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      16/99      3.95G    0.04357    0.01236    0.01773          7        640: 100% 14/14 [00:02<00:00,  5.21it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.34it/s]
                   all         60         61      0.722      0.869      0.919      0.655

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      17/99      3.95G     0.0419    0.01434    0.01667          9        640: 100% 14/14 [00:03<00:00,  4.56it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.59it/s]
                   all         60         61      0.639      0.911      0.776      0.514

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      18/99      3.95G     0.0397    0.01193    0.01457          5        640: 100% 14/14 [00:03<00:00,  4.46it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.75it/s]
                   all         60         61      0.846      0.949      0.927      0.665

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      19/99      3.95G    0.03734    0.01239    0.01386          4        640: 100% 14/14 [00:02<00:00,  4.73it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.91it/s]
                   all         60         61      0.832      0.955      0.946      0.666

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      20/99      3.95G    0.03805     0.0128    0.01334          8        640: 100% 14/14 [00:02<00:00,  4.96it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.12it/s]
                   all         60         61      0.935      0.885       0.97       0.66

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      21/99      3.95G    0.03613    0.01142    0.01248          2        640: 100% 14/14 [00:03<00:00,  3.80it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.45it/s]
                   all         60         61       0.93      0.952      0.976       0.65

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      22/99      3.95G    0.03773     0.0122    0.01247          7        640: 100% 14/14 [00:02<00:00,  4.97it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.55it/s]
                   all         60         61      0.822      0.932      0.926      0.615

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      23/99      3.95G    0.03499    0.01065    0.01126          2        640: 100% 14/14 [00:02<00:00,  5.13it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.50it/s]
                   all         60         61      0.829      0.925      0.956      0.684

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      24/99      3.95G    0.03575    0.01104    0.01251          4        640: 100% 14/14 [00:03<00:00,  3.89it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.01it/s]
                   all         60         61      0.925      0.958      0.978      0.773

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      25/99      3.95G    0.03357    0.01129   0.009865          8        640: 100% 14/14 [00:03<00:00,  4.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.03it/s]
                   all         60         61      0.762      0.872      0.913      0.582

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      26/99      3.95G     0.0335    0.01087    0.01114          3        640: 100% 14/14 [00:03<00:00,  4.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.24it/s]
                   all         60         61      0.884      0.979      0.952      0.623

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      27/99      3.95G    0.03494     0.0101    0.01295          3        640: 100% 14/14 [00:04<00:00,  3.04it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.20it/s]
                   all         60         61      0.942      0.962      0.972      0.659

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      28/99      3.95G      0.034    0.01073    0.01049          2        640: 100% 14/14 [00:03<00:00,  4.42it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.16it/s]
                   all         60         61       0.92      0.977      0.977      0.647

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      29/99      3.95G    0.03302    0.01073   0.009128          3        640: 100% 14/14 [00:03<00:00,  4.43it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.78it/s]
                   all         60         61      0.973      0.963      0.983       0.72

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      30/99      3.95G    0.03245    0.01007   0.007327          5        640: 100% 14/14 [00:04<00:00,  3.08it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.46it/s]
                   all         60         61      0.953      0.979      0.983      0.704

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      31/99      3.95G    0.03274    0.01121   0.008868          6        640: 100% 14/14 [00:03<00:00,  4.44it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.58it/s]
                   all         60         61       0.98      0.976      0.987      0.797

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      32/99      3.95G    0.03141    0.01097    0.01047          5        640: 100% 14/14 [00:03<00:00,  4.54it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.19it/s]
                   all         60         61      0.962      0.974      0.989      0.734

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      33/99      3.95G    0.03034    0.01084    0.01015          6        640: 100% 14/14 [00:04<00:00,  3.22it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.56it/s]
                   all         60         61      0.869      0.973      0.967       0.72

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      34/99      3.95G    0.02962    0.01013    0.01031          5        640: 100% 14/14 [00:03<00:00,  4.44it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.76it/s]
                   all         60         61      0.972      0.967      0.987      0.776

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      35/99      3.95G    0.03211    0.00934    0.01265          3        640: 100% 14/14 [00:03<00:00,  4.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.29it/s]
                   all         60         61      0.978       0.94      0.983      0.712

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      36/99      3.95G    0.03188     0.0105    0.01126          4        640: 100% 14/14 [00:04<00:00,  3.39it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.35it/s]
                   all         60         61      0.976      0.957      0.981      0.775

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      37/99      3.95G    0.03031   0.009911   0.008885          2        640: 100% 14/14 [00:03<00:00,  4.31it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.23it/s]
                   all         60         61       0.98      0.962       0.98      0.746

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      38/99      3.95G    0.03196    0.01183    0.01094          6        640: 100% 14/14 [00:03<00:00,  4.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.17it/s]
                   all         60         61      0.966      0.935      0.967      0.789

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      39/99      3.95G    0.02892   0.009778   0.009674          5        640: 100% 14/14 [00:04<00:00,  3.45it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.94it/s]
                   all         60         61      0.961      0.958       0.98      0.814

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      40/99      3.95G    0.02981    0.01059   0.008671          8        640: 100% 14/14 [00:03<00:00,  4.29it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.20it/s]
                   all         60         61      0.962      0.968      0.981      0.791

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      41/99      3.95G    0.02626    0.01007   0.007103          8        640: 100% 14/14 [00:03<00:00,  4.46it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.71it/s]
                   all         60         61      0.933      0.955       0.98      0.821

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      42/99      3.95G    0.02805   0.009351   0.006896          3        640: 100% 14/14 [00:03<00:00,  3.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.97it/s]
                   all         60         61      0.961      0.981      0.988      0.757

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      43/99      3.95G      0.029    0.01033   0.007909          3        640: 100% 14/14 [00:03<00:00,  4.22it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.09it/s]
                   all         60         61      0.953      0.968      0.992       0.83

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      44/99      3.95G    0.02814     0.0097   0.008948          3        640: 100% 14/14 [00:03<00:00,  3.93it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.53it/s]
                   all         60         61      0.949      0.945      0.978       0.71

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      45/99      3.95G    0.02688   0.009637   0.007706          2        640: 100% 14/14 [00:03<00:00,  3.86it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.74it/s]
                   all         60         61      0.935      0.968      0.976      0.808

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      46/99      3.95G    0.02621    0.01023   0.007679          6        640: 100% 14/14 [00:03<00:00,  4.43it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.85it/s]
                   all         60         61      0.948      0.976      0.979       0.81

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      47/99      3.95G    0.02414   0.008899   0.006105          5        640: 100% 14/14 [00:03<00:00,  3.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.68it/s]
                   all         60         61      0.975      0.976      0.987      0.839

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      48/99      3.95G    0.02671   0.009037   0.005851          4        640: 100% 14/14 [00:03<00:00,  4.46it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.97it/s]
                   all         60         61      0.974      0.981      0.986      0.787

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      49/99      3.95G    0.02856    0.01043   0.006366          7        640: 100% 14/14 [00:03<00:00,  4.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.04it/s]
                   all         60         61      0.972      0.981      0.987      0.801

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      50/99      3.95G    0.02725   0.009708   0.007638          6        640: 100% 14/14 [00:03<00:00,  3.55it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.39it/s]
                   all         60         61      0.982      0.975      0.984      0.813

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      51/99      3.95G    0.02611    0.01009   0.005648          6        640: 100% 14/14 [00:03<00:00,  4.29it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.69it/s]
                   all         60         61      0.983      0.963      0.982      0.803

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      52/99      3.95G    0.02857   0.009762   0.005607          3        640: 100% 14/14 [00:03<00:00,  4.40it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.77it/s]
                   all         60         61       0.96      0.981      0.983      0.802

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      53/99      3.95G     0.0248   0.009939   0.004675          4        640: 100% 14/14 [00:04<00:00,  3.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.77it/s]
                   all         60         61      0.955      0.981      0.989      0.854

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      54/99      3.95G    0.02624    0.01007   0.006999          8        640: 100% 14/14 [00:03<00:00,  4.29it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.26it/s]
                   all         60         61      0.956      0.981      0.986      0.819

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      55/99      3.95G    0.02525   0.009409   0.008536          5        640: 100% 14/14 [00:03<00:00,  4.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.00it/s]
                   all         60         61      0.981      0.975      0.992      0.831

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      56/99      3.95G    0.02489   0.009402   0.006102          3        640: 100% 14/14 [00:04<00:00,  3.03it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.39it/s]
                   all         60         61      0.982      0.981      0.992      0.826

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      57/99      3.95G     0.0234   0.008755   0.007204          5        640: 100% 14/14 [00:03<00:00,  4.27it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.16it/s]
                   all         60         61      0.981      0.981      0.991      0.836

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      58/99      3.95G    0.02448   0.009355   0.007075          6        640: 100% 14/14 [00:03<00:00,  4.25it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.21it/s]
                   all         60         61       0.99      0.955      0.985      0.813

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      59/99      3.95G    0.02344   0.008701    0.00478          5        640: 100% 14/14 [00:04<00:00,  3.09it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.00it/s]
                   all         60         61      0.988      0.961      0.978      0.805

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      60/99      3.95G    0.02396   0.008579   0.004608          5        640: 100% 14/14 [00:03<00:00,  4.27it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.83it/s]
                   all         60         61      0.955      0.981      0.986       0.82

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      61/99      3.95G    0.02351   0.008961   0.005946          3        640: 100% 14/14 [00:03<00:00,  4.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.68it/s]
                   all         60         61      0.981      0.958      0.993      0.852

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      62/99      3.95G    0.02486   0.009275   0.006086          5        640: 100% 14/14 [00:04<00:00,  3.42it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.60it/s]
                   all         60         61      0.972      0.977      0.994      0.855

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      63/99      3.95G     0.0223    0.00834   0.004183          6        640: 100% 14/14 [00:03<00:00,  4.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.07it/s]
                   all         60         61      0.979      0.977      0.994      0.867

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      64/99      3.95G    0.02428    0.01038   0.005671          5        640: 100% 14/14 [00:03<00:00,  4.05it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.84it/s]
                   all         60         61      0.989      0.963      0.993      0.848

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      65/99      3.95G    0.02386   0.008211   0.005882          3        640: 100% 14/14 [00:03<00:00,  3.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.78it/s]
                   all         60         61       0.98      0.981      0.994      0.848

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      66/99      3.95G    0.02123   0.008648   0.006065          5        640: 100% 14/14 [00:03<00:00,  4.18it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.20it/s]
                   all         60         61      0.979      0.981      0.993      0.873

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      67/99      3.95G    0.02145   0.008868   0.003726          4        640: 100% 14/14 [00:03<00:00,  3.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.38it/s]
                   all         60         61      0.986      0.979      0.993      0.884

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      68/99      3.95G    0.02304   0.009829   0.004169          5        640: 100% 14/14 [00:03<00:00,  4.22it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.88it/s]
                   all         60         61      0.987      0.981      0.993      0.867

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      69/99      3.95G    0.02135   0.008979   0.003674          4        640: 100% 14/14 [00:03<00:00,  4.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.22it/s]
                   all         60         61      0.983      0.981      0.994      0.874

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      70/99      3.95G    0.02198   0.008713   0.004601          6        640: 100% 14/14 [00:04<00:00,  3.15it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.69it/s]
                   all         60         61      0.986      0.986      0.995      0.869

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      71/99      3.95G    0.02208   0.008889   0.004078          7        640: 100% 14/14 [00:03<00:00,  4.16it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.76it/s]
                   all         60         61      0.988      0.986      0.995      0.874

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      72/99      3.95G    0.02379   0.009007   0.005575          5        640: 100% 14/14 [00:03<00:00,  4.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.50it/s]
                   all         60         61      0.986      0.986      0.995      0.833

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      73/99      3.95G    0.02254   0.008928   0.003659          3        640: 100% 14/14 [00:04<00:00,  2.96it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.17it/s]
                   all         60         61       0.97      0.988      0.994      0.855

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      74/99      3.95G    0.02294   0.008651   0.005888          4        640: 100% 14/14 [00:03<00:00,  4.24it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.24it/s]
                   all         60         61      0.974      0.986      0.995      0.876

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      75/99      3.95G    0.02076   0.008877   0.007327          3        640: 100% 14/14 [00:03<00:00,  4.10it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.26it/s]
                   all         60         61      0.972      0.988      0.994      0.879

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      76/99      3.95G    0.02037    0.00873   0.003789          4        640: 100% 14/14 [00:04<00:00,  3.10it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.40it/s]
                   all         60         61      0.981      0.984      0.995      0.882

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      77/99      3.95G    0.02135   0.008944   0.004613          5        640: 100% 14/14 [00:03<00:00,  4.18it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.76it/s]
                   all         60         61      0.982      0.989      0.995      0.862

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      78/99      3.95G    0.02185   0.009339   0.005123          5        640: 100% 14/14 [00:03<00:00,  3.98it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.71it/s]
                   all         60         61      0.989      0.986      0.995      0.832

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      79/99      3.95G    0.02089   0.009026   0.002751          7        640: 100% 14/14 [00:04<00:00,  3.40it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.47it/s]
                   all         60         61      0.989      0.986      0.995      0.892

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      80/99      3.95G    0.01956   0.007682   0.004635          4        640: 100% 14/14 [00:03<00:00,  3.96it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  3.11it/s]
                   all         60         61      0.988      0.986      0.995      0.872

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      81/99      3.95G    0.01857   0.008681    0.00497          6        640: 100% 14/14 [00:04<00:00,  3.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.42it/s]
                   all         60         61      0.987      0.987      0.995      0.887

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      82/99      3.95G    0.01914   0.008794   0.003576          4        640: 100% 14/14 [00:03<00:00,  4.17it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.46it/s]
                   all         60         61      0.988      0.988      0.995      0.887

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      83/99      3.95G    0.01768   0.007796   0.004821          2        640: 100% 14/14 [00:03<00:00,  4.17it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.48it/s]
                   all         60         61      0.988      0.988      0.995       0.88

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      84/99      3.95G    0.01843   0.008085   0.005346          4        640: 100% 14/14 [00:04<00:00,  2.88it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.39it/s]
                   all         60         61       0.99      0.985      0.995      0.883

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      85/99      3.95G    0.01884   0.008414   0.004228          8        640: 100% 14/14 [00:03<00:00,  3.91it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.91it/s]
                   all         60         61      0.994      0.985      0.995      0.897

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      86/99      3.95G    0.01833   0.008805   0.003613          8        640: 100% 14/14 [00:03<00:00,  4.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.12it/s]
                   all         60         61      0.992      0.985      0.995      0.892

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      87/99      3.95G    0.01772    0.00785   0.004006          6        640: 100% 14/14 [00:04<00:00,  3.07it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.50it/s]
                   all         60         61      0.993      0.985      0.995      0.893

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      88/99      3.95G     0.0167   0.007558   0.003798          7        640: 100% 14/14 [00:03<00:00,  4.13it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.84it/s]
                   all         60         61      0.992      0.985      0.995      0.897

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      89/99      3.95G    0.01688   0.007463   0.002992          3        640: 100% 14/14 [00:03<00:00,  3.86it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.83it/s]
                   all         60         61      0.991      0.986      0.995      0.894

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      90/99      3.95G    0.01909   0.009185   0.004256          6        640: 100% 14/14 [00:04<00:00,  3.49it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.38it/s]
                   all         60         61       0.99      0.987      0.995      0.906

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      91/99      3.95G    0.01939   0.008863   0.002903          7        640: 100% 14/14 [00:03<00:00,  4.11it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.60it/s]
                   all         60         61       0.99      0.987      0.995      0.896

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      92/99      3.95G     0.0184   0.008592   0.003149          8        640: 100% 14/14 [00:04<00:00,  3.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.34it/s]
                   all         60         61      0.989      0.988      0.995      0.913

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      93/99      3.95G    0.01698   0.008174   0.003008          8        640: 100% 14/14 [00:03<00:00,  3.99it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.72it/s]
                   all         60         61      0.988      0.988      0.995      0.912

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      94/99      3.95G    0.01658   0.008421   0.003526          6        640: 100% 14/14 [00:03<00:00,  4.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.79it/s]
                   all         60         61      0.987      0.989      0.995      0.923

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      95/99      3.95G    0.01748   0.007642   0.003181          2        640: 100% 14/14 [00:04<00:00,  3.05it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.35it/s]
                   all         60         61      0.988      0.988      0.995      0.911

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      96/99      3.95G    0.01693   0.008137   0.003728          3        640: 100% 14/14 [00:03<00:00,  4.28it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.18it/s]
                   all         60         61      0.988      0.988      0.995      0.925

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      97/99      3.95G    0.01633   0.007992   0.002775          4        640: 100% 14/14 [00:03<00:00,  3.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.49it/s]
                   all         60         61      0.989      0.987      0.995      0.918

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      98/99      3.95G    0.01614   0.007345   0.003309          4        640: 100% 14/14 [00:03<00:00,  3.63it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.46it/s]
                   all         60         61      0.988      0.988      0.995      0.928

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      99/99      3.95G    0.01549   0.008223   0.003173          4        640: 100% 14/14 [00:03<00:00,  4.07it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:00<00:00,  2.75it/s]
                   all         60         61      0.988      0.988      0.995       0.92

100 epochs completed in 0.135 hours.
Optimizer stripped from runs/train/exp2/weights/last.pt, 14.4MB
Optimizer stripped from runs/train/exp2/weights/best.pt, 14.4MB

Validating runs/train/exp2/weights/best.pt...
Fusing layers... 
Model summary: 157 layers, 7020913 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 2/2 [00:01<00:00,  1.07it/s]
                   all         60         61      0.988      0.988      0.995      0.928
                  AK47         60         12      0.973          1      0.995      0.995
                   AWP         60         14      0.993          1      0.995      0.796
                   MP9         60         13          1      0.951      0.995      0.938
                   USP         60         22      0.986          1      0.995      0.982
Results saved to runs/train/exp2
  ```
</details>

### Evidências do treinamento

![results](https://user-images.githubusercontent.com/114018304/236247740-3ef31936-4aa7-4cbf-98d2-dc1d60cfe94b.png)
![teste](https://user-images.githubusercontent.com/114018304/236247306-a0def0b3-06ff-4bfa-8936-7713b682524d.png)
![precision curve](https://user-images.githubusercontent.com/114018304/236247589-63e0f1d5-beb0-4d42-b9ef-be3ce949b4b0.png)

## Roboflow

https://universe.roboflow.com/cesar-school-20222/dataset_atividade_talescarvalho/dataset/1
