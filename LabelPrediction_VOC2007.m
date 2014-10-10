clear; clc; load('datasets/VOC2007.mat');
X_trn = VOC2007_TrainX; X_tst = VOC2007_TestX;

for k=1:21
  Y_trn = VOC2007_TrainY(:,k); 
  Y_tst = VOC2007_TestY(:,k);
  ratio = size(find(Y_trn==1),1)/size(find(Y_trn==0),1);
  model = train(Y_trn,sparse(double(X_trn)),['-s 0 -w1 ' num2str(50*round(1/ratio))]);
  [Y_hat,Accu,Deci] = predict(Y_tst,sparse(double(X_tst)),model);
  VOC2007_PreY(:,k) = Y_hat; AP(k) = Accu(1);
  fscore(Y_hat,Y_tst);
end

mean(AP)


