imdb_train = imdb_from_voc('datasets/VOCdevkit2012', 'train', '2012');
imdb_val = imdb_from_voc('datasets/VOCdevkit2012', 'val', '2012');
rcnn_make_window_file(imdb_train, 'external/caffe/examples/pascal-finetuning');
%rcnn_make_window_file(imdb_val, 'external/caffe/examples/pascal-finetuning');

rcnn_exp_cache_features('train','2012');   % chunk1
rcnn_exp_cache_features('val','2012');     % chunk2
rcnn_exp_cache_features('test_1','2012');  % chunk3
rcnn_exp_cache_features('test_2','2012');  % chunk4
test_results = rcnn_exp_train_and_test('2012')
