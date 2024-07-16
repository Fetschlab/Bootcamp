function [b,err] = Dots_fitDDM_1D_noConf(guess,stimval,choice,rt)

% fit choice+RT data to simple 1D DDM
% CF circa 2013
% see Shadlen et al, 2006 book chapter

% stimval: stimulus strength (eg coh), on a signed axis
% choice: 0 (left) or 1 (right)

% guess: initial values for the three params, i.e.:
% k : coefficient converting stimulus strength to units of momentary evidence
% B : height of the bound, or threshold, for decision termination
% Tnd : non-decision time (ms)


% mean of momentary evidence is k*stimval
% standard deviation of momentary evidence is assumed to be 1    
us = unique(stimval);
pRight = nan(length(us),1);
pRight_se = nan(length(us),1);
RTmean = nan(length(us),1);
RTse = nan(length(us),1);
remove = false(length(us),1);
for s = 1:length(us)    
    I = stimval==us(s);
    % remove stimvals with too few repeats
    if sum(I) < 20
        remove(s) = true;    
    else
        pRight(s) = sum(choice(I)==1) / sum(I); % 1 is rightward
        pRight_se(s) = sqrt(pRight(s)*(1-pRight(s)) / sum(I)); % formula for standard error of a proportion
        RTmean(s) = nanmean(rt(I));
        RTse(s) = nanstd(rt(I))/sqrt(sum(~isnan(rt(I))));
    end
end
removeStimvals = isnan(pRight);
removeTrials = ismember(stimval,us(remove));

pRight(removeStimvals)=[];
pRight_se(removeStimvals)=[];
RTmean(removeStimvals)=[];
RTse(removeStimvals)=[];
us(removeStimvals) = [];

% trial by trial
choice(removeTrials) = [];
stimval(removeTrials) = [];
rt(removeTrials) = [];


% for debugging/teaching:
% initialErr = err_fcn_combined(x,choice,stimval,RTmean,RTse);


%% fitting
[b,err,~,~] = fminsearch(@(x) err_fcn_combined(x,choice,stimval,RTmean,RTse), guess);


%% plot the fits vs the data

% temp, no actual fitting, just plot w a given set of params:
% b = guess;

% to generate smooth curves, make a new stimval axis:
g_stimval = us(1):0.01:us(end);

% Pright and RT according to the best-fit model:
meanPr_model = 1 ./ (1 + exp(-2*b(1)*b(2)*g_stimval));
meanRT_model = b(2)./(b(1)*g_stimval) .* tanh(b(1)*b(2)*g_stimval) + b(3);
meanRT_model(g_stimval==0) = b(2)^2 + b(3);

figure(1001); set(gcf,'Color',[1 1 1],'Position',[600 500 450 700],'PaperPositionMode','auto'); clf;
subplot(2,1,1);
errorbar(us,pRight,pRight_se,'bo'); hold on;
plot(g_stimval,meanPr_model,'b-');
xlabel('Motion strength (%coh)');
ylabel('Proportion rightward choices');

subplot(2,1,2);
errorbar(us,RTmean,RTse,'ro'); hold on;
plot(g_stimval,meanRT_model,'r-');
xlabel('Motion strength (%coh)');
ylabel('Reaction time (ms)');


end




%%
function err = err_fcn_combined(x, choice, stimval, meanRT, ~)
    

% 3 params: k, B, Tnd
k = x(1);
B = x(2);
Tnd = x(3);

 % sometimes ones and zeros are not treated as logicals, so force them to be
choice=logical(choice);

% calculate expected probability of rightward choice given model params
% Eq. from Shadlen et al. 2006 book chapter:
Pr_model = 1 ./ (1 + exp(-2*k*B*stimval));

% expected mean RT, also from Shadlen et al. 2006
meanRT_model = B./(k*unique(stimval)) .* tanh(k*B*unique(stimval)) + Tnd;
meanRT_model(unique(stimval)==0) = B^2 + Tnd; % limit as stimval->0


% likelihood is the expected Pright for trials with a rightward choice,
% plus 1-Pright for trials with a leftward choice. The joint probability
% would be computed as the product, so total logL is the sum of the logs
LL_choice = sum(log(Pr_model(choice))) + sum(log(1-Pr_model(~choice)));


% *****************************
% % Alternatively, you can compute LL on the proportions, not individual
% % trials. In general this is not ideal, but it may be necessary in this
% % case because we're fitting the mean RTs and combining LL for RT and
% % choice (see below)
% 
us = unique(stimval); % leave these lines uncommented, as some are used below
for s = 1:length(us)
    r(s) = sum(choice(stimval==us(s)));
    n(s) = sum(stimval==us(s));
end
% 
% % Palmer et al 2005, eq. 4:
% meanPr_model = 1 ./ (1 + exp(-2*k*B*unique(stimval)));
% LL_choice = sum(log( (factorial(n) ./ (factorial(r).*factorial(n-r))) .* meanPr_model'.^(r) .* (1-meanPr_model').^(n-r) ));
%     % ^ n's are generally too big for explicit factorial; approximate with gammaln
% % LL_choice = sum( gammaln(n+1) - gammaln(r+1) - gammaln(n-r+1) + r.*log(meanPr_model'+eps) + (n-r).*log(1-meanPr_model'+eps) );
%     % ^ comment out for now, see below
% *****************************

% With choices, we could get away with a relatively simple calculation of
% the likelihood because binary outcomes with some probability have a known
% distribution, called the Bernoulli distribution (a special case of the
% binomial distribution for n = 1).

% For RT it's harder because we don't know the underlying distribution of
% the random variable for which each RT is a realization. The sampling
% distribution of *mean* RT is, conveniently, Gaussian, by the central
% limit theorem. So we'll use the Gaussian approximation to calculate
% likelihood of mean RTs under the model params.

% Palmer et al 2005, Eq. 3 & A.34+A.35 (assume zero variance in Tr aka Tnd)
mu = abs(k*us); % this is mu-prime in Palmer
VarRT = (B*tanh(B*mu) - B*mu.*sech(B*mu)) ./ mu.^3; 
VarRT(us==0) = 2/3 * B^4; % Eq. limiting case for coh=0
sigmaRT = sqrt(VarRT./n'); % equation in text above Eq 3


% either way, LL is calculated with the Gaussian equation (Palmer Eq 3):
minP = 1e-300;
L_RT = 1./(sigmaRT*sqrt(2*pi)) .* exp(-(meanRT_model - meanRT).^2 ./ (2*sigmaRT.^2));
    % kluge to remove zeros, which are matlab's response to  exp(-[too large a number])
    % otherwise the log will give you infinity
    L_RT(L_RT==0) = minP;
LL_RT = sum(log(L_RT));

% Lastly, although we want to maximize the likelihood, optimization
% algorithms need something to minimize, not maximize. So we take the
% negative LL (because arg_min(-x) = arg_max(x)).
err = -(LL_choice + LL_RT);

end


