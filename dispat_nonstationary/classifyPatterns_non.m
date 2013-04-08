function [X, Y, K, idx, prototype, sigma, MDS, Locdb] = classifyPatterns_non(out,par)

% out : indicates the training image
% Pat : dimensio of the pattern template(better be odd value)
% m   : number of patterns to be skipped
% m1  : number of multiple-grids for this analysis
% way : indicates the distance function used for similarity measurement
% MDS : indicates the number of dimensions of MDS space
% clus: indicates the number of clusters for classification


global temp;global Pat;


% profile on

Pat  = par.Pat;
Patz = par.Patz;
Dim  = par.Dim;
Dimz = par.Dimz;
m    = par.m;
m1   = par.m1;
way  = par.way;
MDS  = par.MDS;
clus = par.clus;


%----------------------
bAskInputs = false;
bFastMDS   = true;
%----------------------




%% Training image 
%__________________________________________________________________________


% Number of patterns : disDim*disDim
disDim  = ceil((Dim  - (1+(Pat -1)*m1) + 1)/m);
disDimz = ceil((Dimz - (1+(Patz-1)*m1) + 1)/m);
fprintf('\n\nNext Phase\n------------------------------------------------\n');
fprintf('Patterns retained for analysis = %d x %d x %d\n', disDim,disDim,disDimz);

%
Locdb=zeros(disDim*disDim*disDimz,3);
%
X=zeros(disDim*disDim*disDimz,Pat^2*Patz);
l=1;
for i=1:disDim
    for j=1:disDim
        for k=1:disDimz
            %X((i-1)*disDim+j,:)=reshape(out(i:i+Pat-1,j:j+Pat-1),1,Pat^2);
            wx = 1+m*(i-1):m1:1+m*(i-1)+(Pat -1)*m1;
            wy = 1+m*(j-1):m1:1+m*(j-1)+(Pat -1)*m1;
            wz = 1+m*(k-1):m1:1+m*(k-1)+(Patz-1)*m1;
           % % The "if" below is to delete completely empty patterns from calculations 
           % % you should also change to X initialization to X=[];
           % if sum(sum(out(wx,wy)))~=0
                X(l,:)=reshape(out(wx,wy,wz),1,Pat^2*Patz);
                %
                Locdb(l,:)=[wx(1),wy(1),wz(1)];
                %
                l=l+1;
           % end
        end
    end
end



%% Dissimilarity Matrix
%__________________________________________________________________________


disDim2=size(X,1);
if ~bFastMDS
    disMat = zeros(disDim2);
end

if bAskInputs
    disp('Which dissimilarity distance method ? ') ;
    disp(' 1 . Euclidean distance ');
    disp(' 2 . Hausdorff distance ');
    disp(' 3 . Modified Hausdorff distance ');
    disp(' 4 . Hough Transform dissimilarity distance');
    disp(' 5 . Cosine distance');
    disp(' 6 . Radon Transform dissimilarity distance');
    disp(' 7 . distance Transform ');
    disp(' 8 . distance Transform (proximity)');

    way = input(' ');
end

fprintf('Dissimilarity Matrix number %d is being built...',way);

switch way
    case 8
        temp=zeros(disDim2,Pat^2*Patz);
        for i=1:disDim2
            DT=bwdist(reshape(X(i,:),Pat,Pat,Patz));
            if DT==Inf
                DT = ones(1,Pat^2*Patz);
            end
            temp(i,:)=reshape(1-DT./max(DT(:)),1,[]);
        end
%         temp = X;
        if ~bFastMDS
            disMat=squareform(pdist(temp)); % only this one is changed for 3D
        end
        
    case 7
        temp=zeros(disDim2,Pat^2);
        for i=1:disDim2
            temp(i,:)=reshape(bwdist(reshape(X(i,:),Pat,Pat)),1,Pat^2);
            if temp(i,:)==Inf
                temp(i,:)= sqrt(2)*(Pat-1).*ones(1,Pat^2);
            end
        end
        if ~bFastMDS
            disMat=squareform(pdist(temp));
        end
    
    case 6
        fprintf('\n Percentage Completed:    ');
        theta=0:359;
        for i=1:disDim2
            perc=round(100*i/disDim2);
            fprintf('\b\b\b\b%3d%%', perc);
            temp(i,:)=reshape(radon(reshape(X(i,:),Pat,Pat),theta),1,[]);
        end
        if ~bFastMDS
            fprintf('\n>Calculating distances...');
            disMat=squareform(pdist(temp));
        end
        
    case 5
        fprintf('>Calculating distances...');
        fprintf('\n Percentage Completed:    ');
        for i=1:disDim2
            for j=i+1:disDim2
                disMat(i,j) = calcDissimilarityCosine(i,j,disDim2,Pat);
                disMat(j,i) = disMat(i,j);
            end
        end
        
     case 4
        fprintf('\n Percentage Completed:    ');
        for i=1:disDim2
            perc=round(100*i/disDim2);
            fprintf('\b\b\b\b%3d%%', perc);
            temp(i,:)=reshape(hough(reshape(X(i,:),Pat,Pat)),1,[]);
        end
        if ~bFastMDS
            fprintf('\n>Calculating distances...');
            disMat=squareform(pdist(temp));
        end

     case 3
        fprintf('>Calculating distances...');
        fprintf('\n Percentage Completed:    ');
        for i=1:disDim2
            for j=i+1:disDim2
                disMat(i,j) = calcDissimilarityModified(i,j,disDim2,Pat);
                disMat(j,i) = disMat(i,j);
            end
        end    

     case 2
        fprintf('>Calculating distances...');
        fprintf('\n Percentage Completed:    ');
        for i=1:disDim2
            for j=i+1:disDim2
                disMat(i,j) = calcDissimilarity(i,j,disDim2,Pat);
                disMat(j,i) = disMat(i,j);
            end
        end
        
        
    case 1     
        if ~bFastMDS
            disMat=squareform(pdist(X));
%         disMat=squareform(pdist(X),'seuclidean');
%         disMat=squareform(pdist(X),'mahalanobis');
%         disMat=squareform(pdist(X),'cityblock');
%         disMat=squareform(pdist(X),'minkowski',3);
%         disMat=squareform(pdist(X),'cosine');
%         disMat=squareform(pdist(X),'correlation');
%         disMat=squareform(pdist(X),'spearman');
%         disMat=squareform(pdist(X),'hamming');
%         disMat=squareform(pdist(X),'jaccard');
%         disMat=squareform(pdist(X),'chebychev');
        end
        
end

fprintf(' Done!\n');

%% Multi Dimensional Scaling
%__________________________________________________________________________

% booleans indicating whether to plot corr. coef. or not
bPlotCorrCoefficient      = true;
bPlotFinalCorrCoefficient = true;


% Select intrinsic dimensionality using PCA
fprintf('Calculating Intrinsic Dimensionality ..........');
S   = slpca(X','preserve',{'Everrit',0});   % other method can be profileML
MDS = S.feadim;
fprintf('\b\b\b\b %3d\n',MDS)


fprintf('MultiDimensional Scaling ......................');

if ~bFastMDS
    [Y1,e] = cmdscale(double(disMat));
end




% Plot EigenValues
% ---------------------------------
if ~bFastMDS
    figure;
    subplot(2,1,1);
    plot(1:min(Pat^2,disDim2),e(1:min(Pat^2,disDim2)));
    graph2d.constantline(0,'LineStyle',':','Color',[.7 .7 .7]);
    axis([1,min(Pat^2,disDim2),min(e),max(e)*0.9]);
    xlabel('Eigenvalue number');
    ylabel('Eigenvalue');
end


% Plot Correlation Coefficients
% ---------------------------------
if ~bFastMDS
    if bPlotCorrCoefficient
        subplot(2,1,2);
        hold on
        corrs=[];
        for i=1:5:min(size(Y1,2),Pat^2)
            CORR = corrcoef(disMat,squareform(pdist(Y1(:,1:i))));
            corrs=[corrs CORR(1,2)];
        end
        plot(1:5:min(size(Y1,2),Pat^2),corrs,'ko-','MarkerEdgeColor','r','MarkerFaceColor','k','MarkerSize',8);
        xlabel('Dimension');
        ylabel('Correlation coefficient');
    end
end


% Do MDS
% ---------------------------------
if bAskInputs
    MDS = input('What is the MDS dimension ? ') ;
end

if ~bFastMDS
    maxerr = max(abs(pdist(X)-pdist(Y1(:,1:MDS))));
    Y = Y1(:,1:MDS);
else
    if size(X,1) < 400
        disMat=squareform(pdist(X));
        [Y1,e] = cmdscale(double(disMat));
        Y = Y1(:,1:MDS);
    else
%         [Y,totaltime] = scmdscale_withDist(X,MDS,200,floor(3*MDS/2),1);
        [Y,totaltime] = scmdscale_withDistAndEigs(X,MDS,500,floor(3*MDS/2),1);
    end
end

% Plot final distances' correlation
% ---------------------------------
if ~bFastMDS
    if bPlotFinalCorrCoefficient
        MDStest = squareform(pdist(Y));
        figure;
        plot(disMat,MDStest,'b.');
        CORR = corrcoef(disMat,MDStest);
        xlabel('Dissimilarity distance');
        ylabel(['Euclidean distance in ',num2str(MDS),' D']);
        title(['Correlation Coefficient = ',num2str(CORR(1,2))]);
    end
end

fprintf(' Done!\n');

%% clustering Algorithms
%__________________________________________________________________________

if par.bUseKernelForClustering
    bShow3DProjection = false;

    % Choosing between clustering methods
    fprintf('Kernel K-means analysis  ......................');

    % way2 = 3;
    MDS1=MDS;

    if bAskInputs
        clus = input('How many clusters ? ') ;
    end

    sigma =0.2*max(max(pdist(Y)));

    % Apply Radial Basis Function Kernel
    % K = kernel(Y', 'rbf', sigma);
    % Faster Kernel about 25%
    K = slkernel_non(Y', 'gauss', sigma, Locdb');
%     K = slkernel(X', 'gauss', sigma);


    % Spectral Clustering to find initial clusters
    % idx = SpectralClustNg( K, clus);


    if bShow3DProjection
        % Define the dimension of the projections on the Feature space
        % e = eig(K);
        %new_dim=length(find(e>1));
        new_dim=3;
        model= kpca(Y', K, struct('ker','rbf','arg',sigma,'new_dim',new_dim));
        Z = kernelproj(Y', model)';
    end

    % Kernel K-Means algorithm with initial points from Spectral Clustering.
    % idx = dualkmeansFast(K,clus,idx);
    idx = dualkmeansFast(K,clus);
        
        % faster alternative is to use hierarchical kmeans and then
        % dualkmeansFast incorporating their labels

    MDS=-1;
    
else
    fprintf('Fast K-means Analysis..........................');
    [dummyCenters, idx] = Fkmeans(Y, clus);
    
end
fprintf(' Done!\n');


%% plotting the results
%__________________________________________________________________________
bShow3DProjection = false;
if bShow3DProjection
    
    figure;
    ptsymb = {'bs','r^','md','go','c+','y*','kv','b>','r<','mp','gh','c.'};
    ptsymb=repmat(ptsymb,1,8); PC=0;


    for i = 1:clus
        clust = find(idx==i);
        plot3(Z(clust,1),Z(clust,2),Z(clust,3),ptsymb{i});
        hold on
    end

    title('KPCA results in 3-D kernel space with centers');
    grid on;
    
end

%% showing average of all clusters patterns
%__________________________________________________________________________


bShowAverageOfClusters = false;

if bShowAverageOfClusters
    calcClusters;           % has not been changed to 3D case
end

prototype = zeros(clus,Pat^2*Patz);
for i = 1:clus
    clust = find(idx==i);
    n=size(clust,1);
    prototype(i,:)=sum(X(clust,:),1)./n;
end


MDS = MDS1;



%% Plotting the clustered categories on training image wrt. their colors
%__________________________________________________________________________

bPlotPrototypeMap = false;
% This whole part has not been adapted to 3D case
if bPlotPrototypeMap
    
    figure;
    Prototype=zeros(Dim,Dim);
    for i = 1:clus
        clust  = find(idx==i);
        Iclust = ceil(clust./disDim);
        Jclust = mod(clust,disDim)+1;
        for j = 1:length(Iclust)
            Prototype(1+m.*(Iclust(j)-1)+(Pat-1)*m1/2,1+m.*(Jclust(j)-1)+(Pat-1)*m1/2)=i;
        end
    end
    imagesc(Prototype);
    axis square

end





end
