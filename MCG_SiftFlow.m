ImageList = importdata('./datasets/SiftFlow/ImageList.txt');
LabelList = importdata('./datasets/SiftFlow/LabelList.txt');

for i = 1:2688
  fprintf('Extract MCG Regions: #%d\n', i);
  im = imread(['./datasets/SiftFlow/Images/' ImageList{i}]);
  candidates_scg = im2mcg(im,'accurate');
  boxes = [candidates_scg.bboxes(:,2),candidates_scg.bboxes(:,1),candidates_scg.bboxes(:,4),candidates_scg.bboxes(:,3)];
  save(['./datasets/SiftFlow/MCG/' LabelList{i}],'boxes','-v7.3');
end
