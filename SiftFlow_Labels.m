LabelList = importdata('./datasets/SiftFlow/LabelList.txt');
SiftFlow_TrainY=zeros(2488,33);

for k = 1:2488
  load(['./datasets/SiftFlow/SemanticLabels/' LabelList{k}]);
  labels=zeros(1,33);
  for i=1:256
    for j=1:256
      if ((S(i,j)~=0) && (labels(S(i,j))~=1))
        labels(S(i,j))=1;
      end
    end
  end
  SiftFlow_TrainY(k,:)=labels;
end
