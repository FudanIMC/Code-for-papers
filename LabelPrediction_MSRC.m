load('MSRC.mat'); X_trn = MSRC_TrainX; X_tst = MSRC_TestX;

for k=1:21
  Y_trn = MSRC_TrainY(:,k); 
  Y_tst = MSRC_TestY(:,k);
  model = train(Y_trn,sparse(double(X_trn)),'-s 0 -w1 10');
  [Y_hat,Accu,Vote] = predict(Y_tst,sparse(double(X_tst)),model);
  AP(k) = Accu(1); MSRC_PreY(:,k) = Y_hat;
  [F_score(k),Precision(k),Recall(k)] = fscore(Y_hat,Y_tst);
end

mean(AP)
