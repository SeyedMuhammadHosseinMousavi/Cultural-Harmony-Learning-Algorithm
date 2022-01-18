function bestfis=hars(fis,data)
%
% Variables
p0=GettingFuzzyParameters(fis);
Problem.CostFunction=@(x) FuzzyCost(x,fis,data);
Problem.nVar=numel(p0);
alpha=1;
VarMin = -10;         % Decision Variables Lower Bound
VarMax = 10;         % Decision Variables Upper Bound
Problem.VarMin=-(10^alpha);
Problem.VarMax=10^alpha;
%
% Harmony Search Parameters
Params.MaxIt = 40;     % Maximum Number of Iterations
Params.HMS = 8;         % Harmony Memory Size
Params.nNew = 3;        % Number of New Harmonies
Params.HMCR = 0.9;       % Harmony Memory Consideration Rate
Params.PAR = 0.1;        % Pitch Adjustment Rate
Params.FW = 0.02*(VarMax-VarMin);    % Fret Width (Bandwidth)
Params.FW_damp = 0.995;              % Fret Width Damp Ratio
%
% Starting Harmony Search Algorithm
results=Runhars(Problem,Params);
%
% Getting the Results
p=results.BestSol.Position.*p0;
bestfis=FuzzyParameters(fis,p);
end
%%----------------------------------------------
function results=Runhars(Problem,Params)
disp('Starting Harmony Search Training');
%------------------------------------------------
% Cost Function
CostFunction=Problem.CostFunction;  
% Number of Decision Variables
nVar=Problem.nVar;   
% Size of Decision Variables Matrixv
VarSize=[1 nVar]; 
% Lower Bound of Variables
VarMin=Problem.VarMin;    
% Upper Bound of Variables
VarMax=Problem.VarMax;      
% Some Change
if isscalar(VarMin) && isscalar(VarMax)
dmax = (VarMax-VarMin)*sqrt(nVar);
else
dmax = norm(VarMax-VarMin);
end
%--------------------------------------------
% Harmony Search Algorithm Parameters
% Maximum Number of Iterations
MaxIt=Params.MaxIt;
% Harmony Memory Size
% nPop=Params.HMS; 
HMS=Params.HMS;
% Number of New Harmonies
nNew=Params.nNew; 
HMCR = 0.9;       % Harmony Memory Consideration Rate
PAR = 0.1;        % Pitch Adjustment Rate
FW = 0.02*(VarMax-VarMin);    % Fret Width (Bandwidth)
FW_damp = 0.995;              % Fret Width Damp Ratio
%------------------------------------------------------
% Second Stage
% Empty Harmony Structure
empty_harmony.Position = [];
empty_harmony.Cost = [];
% Initialize Harmony Memory
HM = repmat(empty_harmony, HMS, 1);
% Create Initial Harmonies
for i = 1:HMS
    HM(i).Position = unifrnd(VarMin, VarMax, VarSize);
    HM(i).Cost = CostFunction(HM(i).Position);
end
% Sort Harmony Memory
[~, SortOrder] = sort([HM.Cost]);
HM = HM(SortOrder);
% Update Best Solution Ever Found
BestSol = HM(1);
% Array to Hold Best Cost Values
BestCost = zeros(MaxIt, 1);
%
%% Harmony Search Algorithm Main Body
%
for it = 1:MaxIt 
    % Initialize Array for New Harmonies
    NEW = repmat(empty_harmony, nNew, 1);
    % Create New Harmonies
    for k = 1:nNew  
        % Create New Harmony Position
        NEW(k).Position = unifrnd(VarMin, VarMax, VarSize);
        for j = 1:nVar
            if rand <= HMCR
                % Use Harmony Memory
                i = randi([1 HMS]);
                NEW(k).Position(j) = HM(i).Position(j);
            end       
            % Pitch Adjustment
            if rand <= PAR
                %DELTA = FW*unifrnd(-1, +1);    % Uniform
                DELTA = FW*randn();            % Gaussian (Normal) 
                NEW(k).Position(j) = NEW(k).Position(j)+DELTA;
            end
        end  
        % Apply Variable Limits
        NEW(k).Position = max(NEW(k).Position, VarMin);
        NEW(k).Position = min(NEW(k).Position, VarMax);
        % Evaluation
        NEW(k).Cost = CostFunction(NEW(k).Position);   
    end 
    % Merge Harmony Memory and New Harmonies
    HM = [HM
        NEW]; %#ok
    % Sort Harmony Memory
    [~, SortOrder] = sort([HM.Cost]);
    HM = HM(SortOrder);
    % Truncate Extra Harmonies
    HM = HM(1:HMS);
    % Update Best Solution Ever Found
    BestSol = HM(1);
    % Store Best Cost Ever Found
    BestCost(it) = BestSol.Cost;
    % Show Iteration Information
    disp(['In Iteration ' num2str(it) ': HS Best Cost Is = ' num2str(BestCost(it))]);
    % Damp Fret Width
    FW = FW*FW_damp;
end
%------------------------------------------------
disp('Harmony Search Algorithm Came To End');
% Store Res
results.BestSol=BestSol;
results.BestCost=BestCost;
% Plot Harmony Search Training Stages
figure
set(gcf, 'Position',  [600, 300, 500, 200])
plot(BestCost,':',...
    'LineWidth',2,...
    'MarkerSize',8,...
    'MarkerEdgeColor','r',...
    'Color',[0.1,0.9,0.1]);
title('Harmony Search Algorithm Training')
xlabel('Harmony Search Iteration Number','FontSize',10,...
       'FontWeight','bold','Color','r');
ylabel('Harmony Search Best Cost Result','FontSize',10,...
       'FontWeight','bold','Color','r');
legend({'Harmony Search Algorithm Train'});
end


