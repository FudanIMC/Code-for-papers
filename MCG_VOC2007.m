clc; clear; List = importdata('./datasets/VOC2007/List.txt');

for i=1:632
  im = imread(['./datasets/VOC2007/Images/' List{i} '.png']); candidates_scg = im2mcg(im,'accurate');
  boxes = [candidates_scg.bboxes(:,2),candidates_scg.bboxes(:,1),candidates_scg.bboxes(:,4),candidates_scg.bboxes(:,3)];
  save(['./datasets/VOC2007/MCG/' List{i} '.mat'],'boxes','-v7.3');
end
