load fisheriris
[ndata,D]=size(meas);
R=randperm(ndata);
Xtest=meas(R(1:0.3*ndata),:);
Xtest_label=species(R(1:0.3*ndata),:);
R(1:0.3*ndata)=[];
Xtrain=meas(R,:);
Xtrain_label=species(R,:);
count=0;
%KNN k-nearest neighbor
knn=ClassificationKNN.fit(Xtrain,Xtrain_label,'NumNeighbors',1);  
Xpredict_label_knn=predict(knn,Xtest);
for i=1:0.3*ndata
    if strcmp(Xpredict_label_knn{i},Xtest_label{i})
        count=count+1;
    end
end
accuracy_knn=count/(ndata*0.3)

%Naive Bayes
count=0;
nb=fitcnb(Xtrain,Xtrain_label);
Xpredict_label_nb=predict(nb,Xtest);
for i=1:0.3*ndata
    if strcmp(Xpredict_label_nb{i},Xtest_label{i})
        count=count+1;
    end
end
accuracy_nb=count/(ndata*0.3)
