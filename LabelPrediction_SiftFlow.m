load('Pool5Feature.mat');
X_trn = SiftFlow_TrainX; 
X_tst=SiftFlow_TestX;

for k=1:33
  Y_trn = SiftFlow_TrainY(:,k); 
  Y_tst = SiftFlow_TestY(:,k);
  model = train(Y_trn,sparse(double(X_trn)));
  [X,Y,Z] = predict(Y_tst,sparse(double(X_tst)),model);
  AP(k) = Y(1);
end

mean(AP)
