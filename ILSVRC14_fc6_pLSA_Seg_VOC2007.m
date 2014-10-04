load('./datasets/VOC2007_Seg.mat'); [wt,td,E] = plsa(VOC2007_Seg(1:20750,:)',300,9); 
load('./datasets/VOC2007.mat'); X_trn = []; X_tst = [];

for i=1:422
  X_trn(i,:) = [VOC2007_TrainX(i,:),VOC2007_TrainX(i,:)*wt];
%  X_trn(i,:) = VOC2007_TrainX(i,:);
%  X_trn(i,:) = VOC2007_TrainX(i,:)*wt;
end

for i=1:210
  X_tst(i,:) = [VOC2007_TestX(i,:),VOC2007_TestX(i,:)*wt];
%  X_tst(i,:) = VOC2007_TestX(i,:);
%  X_tst(i,:) = VOC2007_TestX(i,:)*wt;
end

%for k=1:21
%  Y_trn = VOC2007_TrainY(:,k); 
%  Y_tst = VOC2007_TestY(:,k);
%  model = train(Y_trn,sparse(double(X_trn)));
%  [X,Y,Z] = predict(Y_tst,sparse(double(X_tst)),model);
%  VOC2007_PreY(:,k) = X; AP(k) = Y(1);
%end

%mean(AP)

VOC2007_TrainN = VOC2007_TrainY; threshold = 0.05;

for k=1:21
  for j=1:422
    if (rand(1)<threshold)
      VOC2007_TrainN(j,k) = 1-VOC2007_TrainY(j,k);
    end
  end
end

NL = zeros(1,10);
for i=1:422
  YY = VOC2007_TrainN(i,:); Y = VOC2007_TrainY(i,:);
  changed = size(find(YY~=Y),2); NL(changed+1) = NL(changed+1) + 1;
end

for k=1:20
  Y_trn = VOC2007_TrainN(:,k); 
  Y_tst = VOC2007_TestY(:,k);
  model = train(Y_trn,sparse(double(X_trn)),'-s 2');
  [X,Y,Z] = predict(Y_tst,sparse(double(X_tst)),model);
  VOC2007_PreY(:,k) = X; AP(k) = Y(1);
end

mAP = mean(AP)
NL
