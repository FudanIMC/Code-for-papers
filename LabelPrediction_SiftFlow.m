load('Feature.mat');
X_trn = SiftFlow_TrainX; X_tst = SiftFlow_TestX;
SiftFlow_TrainY(:,16)=[]; SiftFlow_TrainY(:,11)=[]; SiftFlow_TrainY(:,9)=[]; 
SiftFlow_TestY(:,16)=[]; SiftFlow_TestY(:,11)=[]; SiftFlow_TestY(:,9)=[]; 

for k=1:30
  Y_trn = SiftFlow_TrainY(:,k); 
  Y_tst = SiftFlow_TestY(:,k);
  model = train(Y_trn,sparse(double(X_trn)));
  [X,Y,Z] = predict(Y_tst,sparse(double(X_tst)),model);
  AP(k) = Y(1);
end

mean(AP)
