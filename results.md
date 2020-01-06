## Embedding size: 32
### results (trained embedder, distance-classification):

#### evaluation on training set:
python evaluate_embedder_classification.py --batch_size 256 --model learning/cls_2d_close_interp_dev/model_39.pth --dataset data/2d_close_interp/ --distance_cache learning/2d_close_interp_dev_distances.pkl 
(Namespace(batch_size=256, dataset='data/2d_close_interp/', distance_cache='learning/2d_close_interp_dev_distances.pkl', evaluation_set='dev', model='learning/cls_2d_close_interp_dev/model_39.pth', publish_triplets=False, workers=4),)
('Random Seed: ', 214)
("Let's use", 8L, 'GPUs!')
('Loading data into memory...',)
100%|██████████| 16085/16085 [00:05<00:00, 2828.65it/s]
Loading overlap information from cache...
100%|██████████| 16085/16085 [00:07<00:00, 2206.11it/s]
Saving overlap information to cache...
('Finished loading data.',)
62it [00:36,  1.72it/s]
('(Acc: 0.599231, Precision: 0.999683, Recall: 0.198526)',)

#### Held-out evaluation set:
 python evaluate_embedder_classification.py --batch_size 256 --model learning/cls_2d_close_interp_dev/model_39.pth --dataset data/2d_close_interp/ --evaluation_set val
(Namespace(batch_size=256, dataset='data/2d_close_interp/', distance_cache=None, evaluation_set='val', model='learning/cls_2d_close_interp_dev/model_39.pth', publish_triplets=False, workers=4),)
('Random Seed: ', 7531)
("Let's use", 8L, 'GPUs!')
('Loading data into memory...',)
100%|██████████| 4021/4021 [00:01<00:00, 2645.93it/s]
100%|██████████| 4021/4021 [03:41<00:00, 18.16it/s] 
Saving overlap information to cache...
('Finished loading data.',)
15it [00:27,  1.84s/it]
('(Acc: 0.602214, Precision: 0.996207, Recall: 0.205208)',)

### results (trained embedder & classifier):
#### training (last epoch):
('[Epoch 39] Total loss: 5.037687, (Acc: 0.978831, Precision: 0.982662, Recall: 0.974861)',)

#### evaluation on training set:
python evaluate_classification.py --batch_size 128 --model learning/cls_full_2d_close_interp_dev/model_39.pth --evaluation_set dev --dataset data/2d_close_interp/ --distance_cache 2d_close_interp_dev_distances.pkl 
(Namespace(batch_size=128, dataset='data/2d_close_interp/', distance_cache='2d_close_interp_dev_distances.pkl', evaluation_set='dev', model='learning/cls_full_2d_close_interp_dev/model_39.pth', workers=4),)
('Random Seed: ', 8438)
("Let's use", 8L, 'GPUs!')
('Loading data into memory...',)
100%|██████████| 16085/16085 [00:05<00:00, 2853.62it/s]
Loading overlap information from cache...
100%|██████████| 16085/16085 [00:07<00:00, 2220.32it/s]
Saving overlap information to cache...
('Finished loading data.',)
125it [00:35,  3.53it/s]
('(Acc: 0.509969, Precision: 0.627498, Recall: 0.049063)',)
