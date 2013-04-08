clc; close all; 


% find the root path of the method's files
fprintf('Adding Paths to Matlab ........................');
fp      = mfilename('fullpath');
dirName = fileparts(fp);
slash   = strfind(dirName, '\');
dirName = dirName(1:slash(end)-1);


% add the paths to Matlab
addpath(dirName);
addpath([dirName '\dist_pat']);
addpath([dirName '\Training Images']);
addpath([dirName '\HardData']);
addpath([dirName '\Slice 3D']);
addpath([dirName '\scmdscale']);
addpath([dirName '\kdtree']);
addpath([dirName '\Fast Kmeans']);
addpath([dirName '\sltoolbox']);
install_paths([dirName '\sltoolbox']);
addpath([dirName '\stprtool_small']);
addpath([dirName '\stprtool_small\preimage']);
fprintf(' Done!\n');



% -------------------------------------------------------------------------
% Parameter Initialization
% -------------------------------------------------------------------------

% possible choices for Training Image
% ===================================
% channel.dat              101x101
% rectangle.dat            101x101
% categorical.dat          101x101
% circle.dat               101x101
% 3D_69x69x39_binary.dat   
% crack.dat                159x159
% non_stationary.dat       199x199
% meander.dat              111x111
% trenduniSG.dat           150x150
tiName   = 'non_stationary.dat'; 



par.Dim  = 199; % size of 'tiName'
par.Dimz = 1;
par.Pat  = 19;
par.Patz = 1;
par.innerPatch      = 9;
par.innerPatchz     = 1;

par.multipleGrid    = 3;

par.bShowMultiGrids = false;

par.newMG    = true;
%%%%%%%%%%%%%%%%%%%%
par.hardData = false;
%%%%%%%%%%%%%%%%%%%%

w_ssm=0.5;

% histogram transformations
par.bTransCat    = false;
par.bTransCon    = false;

% m : number of patterns to be skipped
par.m    = 4;
par.way  = 8;
par.MDS  = 12;
par.clus = 100;
par.bUseKernelForClustering  = true;

par.bSkipPreviousFrozenNodes = true;

par.bDataEventOptimization   = false;
par.bUseDualTemplate         = false;
par.bPasteDualOnFrozen       = false;


% Load/Save files
par.bLoadVariables = false;
par.bSaveVariables = false;

loop = par.multipleGrid;




% -------------------------------------------------------------------------
% Read Training Image
% -------------------------------------------------------------------------
fprintf('Loading the Training Image ....................');
file_location = [dirName,'\Training Images\',tiName];
[datain, colnames, line1] = loadgeoeas(file_location);
out = geoeas2matlab(datain,[par.Dim par.Dim par.Dimz]);

fprintf(' Done!\n');

%figure;
%imshow(out);

clusterModel = []; Z= []; radonX = [];
patternIdx  = struct;
spaceString = '          '; 


% figure; hold on;


% Change the resolution for initial multiple-resolution setting
if par.newMG
    out_c = out;
    m1 = 1;
    out = imresize(out_c,1/loop); 
    level = graythresh(out);
    out = im2bw(out, level);
    par.Dim = size(out,1);
    par.multipleGrid = 1;
end



% --------------------------------------------------------------------
% construct the realization grid
% --------------------------------------------------------------------
% make a bigger matrix so that the real realization be located in the
% middle of the grid and we have space for the boundry grids
par.szRealization  = par.Dim  + (par.Pat  - 1)*2^(par.multipleGrid-1);
par.szRealizationz = par.Dimz + (par.Patz - 1)*2^(par.multipleGrid-1);
% set the initial value = -1 for the continious case; 0.5 for the binary
% case.
realization       = 0.5*ones(par.szRealization, par.szRealization, par.szRealizationz);





% --------------------------------------------------------------------
% assigning hard data
% --------------------------------------------------------------------
hardData = NaN(size(realization));
if par.hardData
    hardData = readHardData('hardData_neighboring.dat', hardData, par);
    
end




%% ------------------------------------------------------------------------
% For each Coarse Grid Do:
%--------------------------------------------------------------------------
for m1 = loop:-1:1
    
    

    
    % change resolutions in multiple-resolution option
    if par.newMG
        out = imresize(out_c,1/m1); 
        par.Dim = size(out,1);
        level = graythresh(out);
        out = im2bw(out, level);
        if m1 ~= loop
            realization = imresize(realization,(m1+1)/m1);
            realization = +realization;
            level = graythresh(realization);
            realization = im2bw(realization, level);
            par.szRealization  = par.Dim  + (par.Pat  - 1)*2^(par.multipleGrid-1);
            par.szRealizationz = par.Dimz + (par.Patz - 1)*2^(par.multipleGrid-1);
            difd = (size(realization,1)  - par.szRealization)/2;
            realization = realization(difd+1:end-difd,difd+1:end-difd);
        end
        m1 = 1;
    end
    
    
    par.m1 = 2^(m1-1);

    
    % Store the frozen nodes in each coarse simulation
    frozenRealiz = zeros(par.szRealization, par.szRealization, par.szRealizationz);

    
    
    if par.bLoadVariables
        
        fprintf('\n\nLoading variables from savedVar%d.mat ....  ',par.m1);
        load([dirName '\dist_pat\savedVar' num2str(par.m1) '.mat'],'-mat');
        fprintf('Done!\n%s%s%s         ',spaceString,spaceString,spaceString);
        
    else
        % Classify Patterns by dissimilarity Matrix and MDS and & Kernel
        [X, Y, K, idx, prototype, sigma, par.MDS, Locdb] = classifyPatterns_non(out,par);
        
%         [clusterModel, Z] = initializeKernelModel(Y, K, sigma, par);
        
        clusterIdx = cell(par.clus,1);
        clusterIdx = fillClusterIndex(clusterIdx, idx);
%         radonX     = calculateRadonX(X, par.Pat); %not adapted to 3D case
        fprintf('%s%s%s         ',spaceString,spaceString,spaceString);
        
    end
    
    
    if par.bSaveVariables
        save([dirName '\dist_pat\savedVar' num2str(par.m1) '.mat'], 'X', 'Y', 'K', 'idx', 'prototype','sigma', 'clusterModel', 'Z','clusterIdx', 'radonX');
    end
    
    
    
    
    
    
   
    
    
    % Define a random path throught the grid nodes
    par.szCoarseGrid = fix((par.Dim  - 1)/par.m1)+1;
    par.szCoarseGridz= fix((par.Dimz - 1)/par.m1)+1;
    lengthRandomPath = par.szCoarseGrid^2*par.szCoarseGridz;
    randomPath       = randperm(lengthRandomPath);
    
    % Change to subscripts
    [nodeI, nodeJ, nodeK] = ind2sub([par.szCoarseGrid,par.szCoarseGrid,par.szCoarseGridz], randomPath);
    node                  = vertcat(nodeI,nodeJ,nodeK);
    [wx, wy, wz]          = getPatternShape(node, par);
    
    % test
    %for kq =1:2601
    %    if(nodeI(kq)==10 && nodeJ(kq)==10)
    %        aaaaa=kq;
    %    end
    %end
    
    
    % test 'moveHardData'
    % imshow(hardData);
    
    % fix the locations of hardData
    hardDataMoved = moveHardData(hardData, par);
    %if par.m1 == 2, hardDataMoved(67,65) = 0;hardDataMoved(69,65) = 1; end
    realization (~isnan(hardDataMoved)) = hardDataMoved(~isnan(hardDataMoved));
    frozenRealiz(~isnan(hardDataMoved)) = 1;
    
    % ---------------------------------------------------------------------
    % Perform simulation
    % for each node in the random path Do:
    % ---------------------------------------------------------------------
    for i = 1:lengthRandomPath
        
        fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bnode: %5d  Percentage Completed: %3.0f%%',i,100*i/lengthRandomPath);
        
        if par.bSkipPreviousFrozenNodes
            if frozenRealiz(wx(i, (par.Pat+1)/2), wy(i, (par.Pat+1)/2), wz(i, (par.Patz+1)/2)) == 1
                continue
            end
        end
        [dataEvent, status] = getDataEvent(realization, wx(i,:), wy(i,:), wz(i,:));
        
        if par.hardData
            wieghtEvent     = findWeight(hardDataMoved, frozenRealiz, wx(i,:), wy(i,:), wz(i,:));
        else
            wieghtEvent     = ones(1,par.Pat^2*par.Patz);
        end
        
        
        % Check if there is any data conditioning event or not and find the
        % pattern to be pasted on the simulation grid
        rand('twister', sum(100*clock));
        switch status
            case 'empty'
                randIdx    = ceil(size(X,1).*rand(1,1));
                if par.bUseDualTemplate
                    patternIdx = findRangeDualTemplate(randIdx, par);
                end
                Pattern    = X(randIdx,:);
            case 'some'
                % get the location of data event
                dataLoc=[wx(i,1), wy(i,1), wz(i,1)];
                % calculate d_pat and d_loc and d_ns
                idxNumber = findClosestPattern_Non(dataEvent, X, dataLoc, Locdb, wieghtEvent,w_ssm);
                %[Pattern, patternIdx] = findClosestInCluster(dataEvent, X, clusterIdx{idxNumber}, par,radonX, wieghtEvent);
                Pattern    = X(idxNumber,:);
                
                hd_condition = hardDataMoved(wx(i,:), wy(i,:), wz(i,:));
                Pattern(~isnan(hd_condition))=hd_condition(~isnan(hd_condition));
%                 idxNumber = findClosestCluster(dataEvent, prototype);
%                 [Pattern, cluster]   = createPattern(idxNumber, clusterModel, Z, X, Y, idx, Pat, cluster,radonX);
%                 Pattern   = reshape(Pattern, 1, Pat^2);
            case 'full'
                if existNonFrozenNodes(frozenRealiz, wx(i,:), wy(i,:), wz(i,:))
                    dataLoc=[wx(i,1), wy(i,1), wz(i,1)];
                    % calculate d_pat and d_loc and d_ns
                    idxNumber = findClosestPattern_Non(dataEvent, X, dataLoc, Locdb, wieghtEvent,w_ssm);
                    %[Pattern, patternIdx] = findClosestInCluster(dataEvent, X, clusterIdx{idxNumber}, par,radonX, wieghtEvent);
                    Pattern    = X(idxNumber,:);
                    
                    hd_condition = hardDataMoved(wx(i,:), wy(i,:), wz(i,:));
                    Pattern(~isnan(hd_condition))=hd_condition(~isnan(hd_condition));
                    %idxNumber = findClosestCluster(dataEvent, prototype, wieghtEvent);
                    %[Pattern, patternIdx] = findClosestInCluster(dataEvent, X, clusterIdx{idxNumber}, par,radonX, wieghtEvent);
                else
                    continue
                end
        end
        
        % for simplicity. use values instead of variables.
        
        
        

        % Paste the pattern on simulation grid and updates frozen nodes
        if par.bUseDualTemplate
            [realization, frozenRealiz] = pastePattern(Pattern, wx(i,:), wy(i,:), wz(i,:), realization, frozenRealiz, par, out(patternIdx.x,patternIdx.y, patternIdx.z));
        else
            [realization, frozenRealiz] = pastePattern(Pattern, wx(i,:), wy(i,:), wz(i,:), realization, frozenRealiz, par, []);
        end
        
       
        
        
    end
    
    
    % show results at each multiple-grid
    if par.bShowMultiGrids
        if par.Dimz > 1 %3D case
            slice3d(realization((par.szRealization-par.Dim)/2+1:(par.szRealization-par.Dim)/2+par.Dim,(par.szRealization-par.Dim)/2+1:(par.szRealization-par.Dim)/2+par.Dim));
            pause(0.1);
        else
            colormap(copper);   %2D
            imagesc(realization((par.szRealization-par.Dim)/2+1:(par.szRealization-par.Dim)/2+par.Dim,(par.szRealization-par.Dim)/2+1:(par.szRealization-par.Dim)/2+par.Dim));
            axis square off;
            drawnow expose;
            pause(0.1);
        end
    end
    
    % to do TRANSCAT in the penultimate multigrids
    if par.bTransCat || par.bTransCon
        if par.m1 == 2
            limits  = (par.szRealization  -par.Dim )/2;
            limitsz = (par.szRealizationz -par.Dimz)/2;
            if ~par.bUseDualTemplate
                real     = realization(limits+1:limits+par.Dim , limits+1:limits+par.Dim , limitsz+1:limitsz+par.Dimz);
                harddata = hardData   (limits+1:limits+par.Dim , limits+1:limits+par.Dim , limitsz+1:limitsz+par.Dimz);
                if par.bTransCon
                    real = histeq(real(1:par.m1:end,1:par.m1:end,1:par.m1:end),hist(out(:),par.Dim^2*par.Dimz));
                else
                    real = transcat(real(1:par.m1:end,1:par.m1:end,1:par.m1:end), out, harddata(1:par.m1:end,1:par.m1:end,1:par.m1:end), par, 1);
                end
                realization(limits+1:par.m1:limits+par.Dim , limits+1:par.m1:limits+par.Dim , limitsz+1:par.m1:limitsz+par.Dimz) = real;
            else
                real     = realization(limits+1:limits+par.Dim , limits+1:limits+par.Dim , limitsz+1:limitsz+par.Dimz);
                harddata = hardData   (limits+1:limits+par.Dim , limits+1:limits+par.Dim , limitsz+1:limitsz+par.Dimz);
                if par.bTransCon
                    real = histeq(real,hist(out(:),par.Dim^2*par.Dimz));
                else
                    real = transcat(real, out, harddata, par, 1);
                end
                realization(limits+1:limits+par.Dim , limits+1:limits+par.Dim , limitsz+1:limitsz+par.Dimz) = real;
            end
        end
    end
    
    
end




% crop the realization to its true dimensions
limits  = (par.szRealization  -par.Dim )/2;
limitsz = (par.szRealizationz -par.Dimz)/2;
realization = realization(limits+1:limits+par.Dim , limits+1:limits+par.Dim , limitsz+1:limitsz+par.Dimz);
% hardData    = hardData   (limits+1:limits+par.Dim , limits+1:limits+par.Dim , limitsz+1:limitsz+par.Dimz);



% TRANSCAT (Transformation of categorical proportions of realization to TI proportions)
if par.bTransCat
    realization_C = transcat(realization, out, hardData, par, 1);
    % show the transformed version of the realization too
    figure; imagesc(realization_C);
    axis square; axis xy off; colormap copper;
end



fprintf('\n\nFinish!\n');


