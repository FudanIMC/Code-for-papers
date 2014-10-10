clear; clc; load('datasets/VOC2007.mat');
X_trn = VOC2007_TrainX; X_tst = VOC2007_TestX;

for k=1:21
  Y_trn = VOC2007_TrainY(:,k); 
  Y_tst = VOC2007_TestY(:,k);
  model = train(Y_trn,sparse(double(X_trn)));
  [X,Y,Z] = predict(Y_tst,sparse(double(X_tst)),model);
  VOC2007_PreY(:,k) = X; AP(k) = Y(1);
  fscore(X,Y_tst);
end

mean(AP)


