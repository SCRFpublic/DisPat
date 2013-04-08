function idxNumber = findClosestPattern_Non(dataEvent, X, dataLoc, Locdb, wieghtEvent, w_ssm)

% This function finds the closest pattern to the dataevent
index1=1:size(X,1);
index_hardData=find(wieghtEvent == 0.5);
if length(index_hardData) ~= 0
    reducedX_hd = X(:, index_hardData);
    reducedDataEvent_hd = dataEvent(1, index_hardData);

    diff1 = sum(abs(bsxfun(@minus, reducedX_hd, reducedDataEvent_hd)),2);
    % for the binary case
    %index1=find(diff1 == 0);
    % for the continious case
    index1=find(diff1/length(index_hardData) <= 0.1);
    % if index1 is empty
    if length(index1)==0
        index1=1:size(X,1);
    end
    
end


%index = find(dataEvent ~= -1);
index = find(dataEvent ~= 0.5);
% only search through Non-NaN s
reducedX = X(index1, index);
reducedDataEvent = dataEvent(1, index);
wieghtEvent      = wieghtEvent(1,index);

% from reducedX, find the rows in which there are the same hard data as in
% the reducedDataEvent


difference1 = sum(bsxfun(@times,abs(bsxfun(@minus, reducedX, reducedDataEvent)),wieghtEvent), 2);
%[dummy, idxNumber] = min(difference);

weightLocation = ones(1, length(dataLoc));
difference3 = sum(bsxfun(@times,abs(bsxfun(@minus, Locdb(index1,:), dataLoc)),weightLocation), 2);

if sum(difference1)~=0
    
    d1_normalize=difference1/(max(difference1)-min(difference1));
else
    d1_normalize=0;
end

d3_normalize=difference3/(max(difference3)-min(difference3));

d=(1-w_ssm)*d1_normalize+w_ssm*d3_normalize;
[dummy, idxNum] = min(d);

idxNumber=index1(idxNum);

%difference4=zeros(size(reducedX,1),1);
%for i= 1:size(reducedX,1)
%    difference4(i) = sqrt(sum((reducedX(i,:)-reducedDataEvent).^2));
%end







%difference2=zeros(size(Locdb,1),1);
%for i= 1:size(Locdb,1)
%    difference2(i) = sqrt(sum((Locdb(i,:)-dataLoc).^2));
%end


end




