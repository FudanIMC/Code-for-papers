clear; close all; clc; load('VOC2007_Regions.mat');
List = importdata('./datasets/VOC2007/List.txt');
caffe('set_device',1); 
rcnn_model = rcnn_create_model(1,'./model-defs/VGG_ILSVRC_19_layers_batch_1_fc6.prototxt', './data/caffe_nets/VGG_ILSVRC_19_layers.caffemodel');
rcnn_model = rcnn_load_model(rcnn_model);
rcnn_model.detectors.crop_mode = 'wrap';
rcnn_model.detectors.crop_padding = 16;
cooc = zeros(20,20); K_means = 7;

for i=1:1%422
  im = imread(['./datasets/VOC2007/Images/' List{i} '.png']); 
  fprintf('Extract %d Train Feature\n',i); imsize = [1 1 size(im,2) size(im,1)];
  load(['./datasets/VOC2007/MCG/' List{i} '.mat']);
  load(['./datasets/VOC2007/MCG_RCNN/' List{i} '.mat']);
  X_tst = feat; Y_tst = ones(1,size(X_tst,1))'; bboxes = []; 
  for k=2:21
    if (VOC2007_TrainY(i,k))
      [X,Y,Z] = predict(Y_tst,sparse(double(X_tst)),SVM_model(k));
      Regions = [boxes,X]; Regions = sortrows(Regions,-5); 
      N_Regions = size(find(Regions(:,5)==1),1);
      bbox = vl_gmm(Regions(1:N_Regions,1:4)',K_means)';
      for j=1:K_means
        dropout_feat(j,:)=rcnn_features(imdropout(im,bbox(j,:)), imsize, rcnn_model);
      end
      [X,Y,Z] = predict(ones(1,K_means)',sparse(double(dropout_feat)),SVM_model(k));
      bboxes(k-1,:)
      for j=2:k-1
        if (VOC2007_TrainY(i,j))
          cooc(j-1,k-1) = cooc(j-1,k-1) + boxoverlap(bboxes(j-1,:),bboxes(k-1,:));
        end
      end
    end
  end
end

LabelCount = zeros(1,4);
for i=1:422
  Current = sum(VOC2007_TrainY(i,:)) - 1;
  if (Current) LabelCount(Current) = LabelCount(Current) + 1; end
end
TrainLabelCount = LabelCount

LabelCount = zeros(1,5);
for i=1:210
  Current = sum(VOC2007_TestY(i,:)) - 1;
  if (Current) LabelCount(Current) = LabelCount(Current) + 1; end
end
TestLabelCount = LabelCount

k=4;
for i=1:422
  Current = sum(VOC2007_TrainY(i,:)) - 1;
  if (Current==k) fprintf('Total %d Labels: #%d\n',k,i); end
end