training=importdata('SPECT_train.txt');
test=importdata('SPECT_test.txt')
%training(find(training==0))=-1;
%test(find(test==0))=-1;
%training data
w=zeros(1,22);
b=0;
count=0;%correct classification
iteration_count=0;%iteration number
sr=0.1;%study rate
fprintf('iteration\t wrong \t\t weight\t\t bias\t\n');

while count ~=size(training,1) 
    %loop stop until no one gets wrong classification or max iterations 
    count=0;
  if iteration_count>10*size(training,1)
      break;
  end
    for i=1:size(training,1)
        count=count+1;
      if training(i,1)*(training(i,2:size(training,2))*w'+b)<=0 %yi*(wi+b<0)if some point gets wrong classification
            w=w+sr*training(i,1)*training(i,2:size(training,2));%update w=w+r*yi*xi
            b=b+sr*training(i,1);%update b=b+r*yi
            iteration_count=iteration_count+1;
            count=count-1;%wrong classification -1
            fprintf('\t%u\t',iteration_count);
            fprintf('\t\t%u\t',i); 
            fprintf('\t(%2.1g,%2.1g)''\t',w);  
            fprintf('\t%4.1g\n',b);         
      end
    end
end

error=0;
%testing data
for i=1:size(test,1)
    x=test(i,2:size(test,2));
    d=dot(w',x)+b;
    if d<=0
        d=0;
    else
        d=1;
    end
    if d ~=test(i,1)
            error=error+1;
    end
end
ErrorRate=error/size(test,1)*100;
CorrectRate=100-ErrorRate;
fprintf('ErrorRate\t  %.2f%%  \n',ErrorRate);
fprintf('CorrectRate\t  %.2f%%',CorrectRate);



            
