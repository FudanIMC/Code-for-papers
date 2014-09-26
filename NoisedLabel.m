function [mAP]=NoisedLabel(theta)

clear; clc; load('Pool5Feature.mat'); X_trn = SiftFlow_TrainX; X_tst = SiftFlow_TestX;
SiftFlow_TrainY(:,16)=[]; SiftFlow_TrainY(:,11)=[]; SiftFlow_TrainY(:,9)=[]; 
SiftFlow_TestY(:,16)=[]; SiftFlow_TestY(:,11)=[]; SiftFlow_TestY(:,9)=[]; 
Change=ones(1,30); Change(25)=0; Change(29)=0; Change(6)=0; Change(14)=0; Change(19)=0;
SiftFlow_TrainN = SiftFlow_TrainY;

for k=1:30
  for j=1:2488
    if (rand(1)<theta)
      SiftFlow_TrainN(j,k) = 1-SiftFlow_TrainN(j,k);
    end
  end
end 

for k=1:30
  Y_trn = SiftFlow_TrainN(:,k); 
  Y_tst = SiftFlow_TestY(:,k);
  model = train(Y_trn,sparse(double(X_trn)),'-s 2');
  [X,Y,Z] = predict(Y_tst,sparse(double(X_tst)),model);
  SiftFlow_PreY(:,k) = X; AP(k) = Y(1);
end

mAP = mean(AP);
