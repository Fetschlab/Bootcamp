% Tutorial on behavioral data analysis/modeling
% CRF : Hopkins-Janelia Bootcamp 2023

% code can be found here: https://github.com/Fetschlab/Bootcamp
% can download as .zip and put in any local folder.

% then change this to your folder, or comment it out:
cd /Users/chris/Documents/Teaching/Bootcamp



%%  1.1 - load and plot some real monkey data

clear; close all
load dots_data

% 'data' is a struct with fields:
% direction : motion direction of the stimulus, 0/180 = right/left
% coherence : motion coherence (proportion of dots moving in the assigned dir)
% scoh : 'signed' coherence, where positive = rightward
% choice : subject's choice, 1=right, 0=left
% RT : reaction time, in seconds
% correct : 1=correct, 0=error

% create indexing (looping) variable
cohs = unique(data.scoh)

% preallocate 
pRight = nan(length(cohs),1);
n = nan(length(cohs),1);
% loop over signed coherence
for c = 1:length(cohs)
    I = data.scoh==cohs(c); % find trials with w this particular coh
    n(c) = sum(I); % how many?
    pRight(c) = sum(I & data.choice==1) / n(c); % proportion of those trials where the choice was "right"
end

% % normally we'd want error bars, so here's the formula for standard error
% % of a proportion (derivation left as an exercise for the reader):
% pRightSE = sqrt( (pRight.*(1-pRight)) ./ n );
% % but it's so tiny with this many trials, we can ignore it for now

figure(1);
set(gcf,'Color',[1 1 1],'Position',[300 500 700 550]);

h1=plot(cohs,pRight,'ko','MarkerSize',9); set(h1,'LineWidth',2);
xlabel('Motion strength (%coh)');
ylabel('Proportion rightward choices');
changeAxesFontSize(gca,15,15);

% plot a small subset of individual trials just for visualization:
hold on;
for n = 1:60
    plot(data.scoh(n)+randn*0.01,data.choice(n),'rx','MarkerSize',12);
                    % ^add some random jitter to reduce overlap
end
legend(h1,'data','Location','Northwest');


%%  1.2 - descriptive/statistical model: logistic regression

% a first step is to quantify how the outcome, or dependent, variable
% (choice) varies as a function of the independent variable (coherence)

% For most outcome variables it is fine to use linear regression: 
% good old y = mx+b. But proportions/probabilities are bounded at 0 and 1,
% so we use logistic regression, log(P/(1-P)) = mx+b
% or more commonly, logit(P) = beta0 + beta1*x + ...

% logit is a synonym for log odds, where odds is P/(1-P)
% (often used in betting, eg "4 to 1 odds" = 0.2 or 20% probability)
% so, logit(P) = log(P/(1-P))

% The inverse of the logit function is the logistic function: 
% P = 1 / (1 + exp(-(beta_0 + beta1*x)))
% So you often see it written this way. This is how you'd actually
% calculate the probabilities given the beta terms.

% The simplest logistic regression model for a choice function
% (aka 'psychometric' function) has two parameters (betas):
% a slope and a bias, equivalent to slope and intercept for linear reg

% The slope term is just a constant multiplied by a single independent
% variable (here, coh), but you can build a logistic model with any number
% of explanatory variables, and their (multiplicative) interactions.
% It is a very flexible and powerful tool, part of a larger family of
% models called *generalized linear models* (GLMs)

% (On your own you can check out any number of video tutorials online)


%%  1.2.1 - likelihood and binomial probability

% The best-fitting model is the one that maximizes the *likelihood* of the
% data given the model parameters

% Likelihood, or P(data | model), is usually calculated by assuming a
% particular form of the probability distribution.

% Since our outcome variable is binary, the relevant distribution is the
% binomial distribution -- and if we are treating each trial as an
% independent event, we use the special case of the binomial dist for n=1,
% known as the Bernoulli distribution.

% For any a particular value of the slope and bias terms, the logistic
% model specifies the probability of a rightward choice (y) given a
% coherence (x).

% Using this model-based probability (P), the Likelihood of
% each trial is simply P if the choice was rightward (1), or 1-P if the
% choice was leftward (0).

% Let's make an initial guess for the parameters and plot the curve atop
% the data points
beta_0 = 0; % assume zero bias
beta1 = 8; % a guess, based on previous experience

% can evaluate the function at any x value (i.e., interpolated or
% extrapolated) not just the values present in the dataset
cohsInterp = -0.51:0.01:0.51;
B = beta_0 + beta1*cohsInterp;
pR = logistic(B);

figure(1);
h2 = plot(cohsInterp,pR,'m--','LineWidth',2);
xlabel('Motion strength (%coh)');
ylabel('Proportion rightward choices');
changeAxesFontSize(gca,15,15);
legend([h1 h2],'data','guess','Location','Northwest');

% calculate the single-trial likelihoods:
pR_model = nan(length(data.choice),1);
Lik = nan(length(data.choice),1);
for t = 1:length(data.choice) % loop over trials
    B = beta_0 + beta1*data.scoh(t);
    pR_model(t) = logistic(B);       
    if data.choice(t)==1
        Lik(t) = pR_model(t);
    else
        Lik(t) = 1-pR_model(t);
    end
end
% (offline coding exercise: rewrite this without using a FOR loop)


% A snapshot of what we've done:
['scoh   choice   pR_model   Likelihood']
[data.scoh(1:10) data.choice(1:10) pR_model(1:10) Lik(1:10)]
% In particular, compare rows 3+4 vs. 5:
% in 5 the coh was positive (right) but monkey chose left, an error that
% was not predicted by the  model (expected pRight for that trial was high)
% Thus, the likelihood is low, i.e. 1-Pr.


% The full likelihood for the  dataset would be the product of all the
% individual likelihoods (joint probability). This calculation is made much
% simpler if we take the log-likelihood. Then we can just sum them up, 
% because the logarithm turns products into sums.
logL_total = sum(log(Lik))


%%  1.3 - optimization

% Now that we know how to calculate the Likelihood, we could write an
% optimization function that iteratively searches over values of slope
% and bias until the (log) Likelihood is maximal -- these would be the
% best-fitting parameters for a given model

% Fortunately Matlab has done that for us in a very simple package, one
% that can be used for much more complicated models than just the
% 2-parameter logistic, so it is worth knowing. It's called glmfit.

X = data.scoh;
y = data.choice;
    % NOTE that we are not curve-fitting the data points (proportions),
    % we are fitting based on likelihood of individual trials.
    % This is more powerful and accurate.
[Beta, ~, stats] = glmfit(X, y, 'binomial');
% (for afficionados: glmfit automatically includes a 'column of zeroes'
% in the design matrix X, to estimate the bias term)


Beta
% Beta is a two-vector with [beta0, beta1] or [bias, slope]
% Notice the bias is almost zero. The slope is in units that are not
% intuitive, because remember it's logit(P) = slope*x + bias
% but it's still true that the higher the number the steeper the slope

%% 1.3.1  plot the fit

% to plot our best fitting model vs. the data points, can use glmval to
% give the model-based probability of rightward given the params:
yVals = glmval(Beta,cohsInterp,'logit');
figure(1); h3 = plot(cohsInterp,yVals,'g-','LineWidth',2);
legend([h1 h2 h3],'data','guess','fit','Location','Northwest');

% pretty good!


%% 2.1 - mechanistic/process models: signal detection theory (SDT)

clear scoh
close all
ntrials = 50000;

% A logistic model is nice but merely descriptive. It describes average
% behavior over ensembles of trials (a given set of independent variables),
% but we want to know how the brain makes a decision on a single trial.
% For this we need a mechanistic or 'process' model.

% In SDT, a stimulus is said to give rise to a noisy 'observation' or
% measurement (in the brain). You can think of it as something like the
% number of spikes in a population of neurons over a given time interval.
% (This makes most sense for detecting a faint signal embedded in noise;
% in the case of discriminating or categorizing stimuli (eg Left/Right)
% it's better to think of the observation as the difference in spike counts 
% between two populations, one selective for Right and the other for Left, 
% as explained in the slides)

% The observation is thus a random variable with some distribution. Let's
% assume Gaussian, with mean proportional to the stimulus strength (coh).
% For simplicity we'll also assume the standard deviation is a constant.

k = 8; % proportionality constant converting our stimulus variable (coh)
       % to the mean of the evidence (obs) distribution; chosen to
       % approximate the data (higher the value, the greater the sensitivity)
sigma = 1; % s.d. of the signal distribution

% plot the obs distributions for a subset of cohs
figure(3); set(gcf,'Color',[1 1 1],'Position',[300 500 1100 400]);
Mu = k*cohs;
dAxis = -8:0.1:8;
for c = 1:2:11
    plot(dAxis,normpdf(dAxis,Mu(c),sigma),'LineWidth',1.5);
    hold on;
end
plot([0 0],[0 0.45],'k--','LineWidth',1.5);
xlabel('Observation');
ylabel('Probability');
changeAxesFontSize(gca,15,15);
legend(num2str(cohs(1:2:11)))



%% 2.1.1  - simulate some trials
scoh = nan(ntrials,1);
obs = nan(ntrials,1);
choice = nan(ntrials,1);
for t = 1:ntrials
    scoh(t) = randsample(cohs,1,'true');
    mu = k*scoh(t);
    obs(t) = normrnd(mu, sigma);
    
    % A defining feature of SDT is the criterion. Whichever side of the
    % criterion the observation falls, that determines the choice.
    % Because coherence is signed, w positive values indicating rightward,
    % the optimal (unbiased) criterion is simply zero: if greater than
    % zero, choose Right (1), else choose Left (0)
    if obs(t)>0
        choice(t)=1;
    else
        choice(t)=0;
    end
end
pRight = nan(length(cohs),1);
n = nan(length(cohs),1);
for c = 1:length(cohs)
    I = scoh==cohs(c);
    n(c) = sum(I);
    pRight(c) = sum(I & choice==1) / n(c);
end
figure(4); set(gcf,'Color',[1 1 1],'Position',[300 500 700 550]);
plot(cohs,pRight,'bo-','LineWidth',1.5);
xlabel('Motion strength (%coh)');
ylabel('Proportion rightward choices');
title('simulated data: SDT');
changeAxesFontSize(gca,15,15);


% For today we'll stop at stimulation, but this model could be fit to data
% and used to generate hypotheses about the neural basis of signal, noise,
% and criterion.

% So although it still seems rather abstract, SDT was among the first
% process models for perception that could be translated into a
% so-called 'linking hypothesis', a key development in the birth of
% systems neuroscience

% [BACK TO SLIDES]



%%  2.2 - mechanistic/process models: drift-diffusion (DDM)

% What if we want to explain more than just the choice?

% Often we measure some other aspect of behavior and ask whether it arises 
% from the same process in the brain. If so, our process model should be
% able to explain both variables. A good example is choice + response time. 

clear; close all
load dots_data

cohs = unique(data.scoh);
pRight = nan(length(cohs),1);
meanRT = nan(length(cohs),1);
n = nan(length(cohs),1);
for c = 1:length(cohs) % loop over signed coherence
    I = data.scoh==cohs(c); % find trials with w this particular coh
    n(c) = sum(I); % how many?
    pRight(c) = sum(I & data.choice==1) / n(c); % proportion of those trials where the choice was "right"
    meanRT(c) = mean(data.RT(I));
    % % again we should always consider (and plot) error/dispersion in data
    % % (standard errors or confidence intervals), which for RT would be:
    % RTse(c) = std(data.RT(I))/sqrt(n(c)); ...
    % % but we'll ignore it for this exercise
end

figure(5); set(gcf,'Color',[1 1 1],'Position',[300 500 450 600]);
title('data: choice + RT');

subplot(2,1,1); plot(cohs,pRight,'bo-');
xlabel('Motion strength (%coh)'); ylabel('Proportion rightward choices');
changeAxesFontSize(gca,15,15);

subplot(2,1,2); plot(cohs,meanRT(:,1),'ro-');
xlabel('Motion strength (%coh)'); ylabel('Reaction time (ms)');
changeAxesFontSize(gca,15,15);


%%  2.2.1 - what kind of process can explain choice + RT?

% SDT has no explicit time dimension and thus cannot explain RT

% So we turn to bounded evidence accumulation,
% e.g. the drift-diffusion model (DDM)

% First, let's simulate it.

% The model has at least two, and usually three parameters 
% (play around with these to see their effects)
k = 0.3; % 'drift rate' or sensitivity term: a constant converting stimulus strength into units of 'momentary' evidence
B = 25; % height of the bound, or threshold, for decision termination
Tnd = 300; % non-decision time: sensory and motor delays distinct from the decision process
% ^ or this can be expressed as a distribution, ie assumed o vary across trials (more natural but adds free param(s))
% and:
% this is sometimes a 4th free parameter but for simplicty we fix it to 1
sigma = 1; % standard deviation of momentary evidence

ntrials = 50000;

data = simDDM_simple_forBootcamp(ntrials,k,sigma,B,Tnd);
% (walk through it, if there's time)



%%  2.3 - a critical step in almost any modeling effort: PARAMETER RECOVERY

% Of course we want to do more than just simulate data that look reasonable
% We want to actually *fit* our process model to the data.  Why?
% (1) to verify that the model is a reasonable one
% (2) to compare goodness-of-fit across different candidate models
% (3) to compare fitted parameters across conditions/subjects and thereby
%     generate hypotheses as to what might differ

% But before we can trust our model fits, we should check that
% our experiment and fitting code are capable, in principle, of correctly
% arriving at the 'ground truth'. To do this we can generate fake data
% by simulating the model (in which case  we know the ground truth, aka the
% generative parameters), then fit the fake data to try and recover
% those params.


% (for more see e.g. Wilson & Collins 2019,
% Ten simple rules for the computational modeling of behavioral data
% https://github.com/AnneCollins/TenSimpleRulesModeling/ )


% As with the logistic, we need an expression for the likelihood: the
% probability of observing the data given a set of model params.

% Choice is still just a binary outcome and hence Bernoulli distributed.
% But what is the expected pRight for each coherence? And there's no
% a priori reason to assert any particular distribution for RT.

% This is where process models differ from descriptive: we need the
% probability density of these variables under the model. This can be 
% generated by simulation, but that's very slow when we want to iterate 
% many times, as we need to do for optimization.

% Luckily smart folks have derived closed-form expressions for pRight (it
% happens to be a logistic function of drift rate times bound!), and
% RT (not intuitively obvious), at least for simple DDMs.

% These are: 
% pRight = 1 / (1 + exp(-2*k*C*B)) % (look familiar?)
% meanRT = B/k*C * tanh(k*C*B)

% See Shadlen et al. 2006 for the derivations.

% to generate our initial guess, randomize the generative parameters
% (pretending we don't know what they were)
guess = [k*(1+randn) B*(1+randn) Tnd*(1+randn)]

% assumes sigma=1
[fit,~] = Dots_fitDDM_1D_noConf(guess,data.scoh,data.choice,round(data.RT*1000)); % convert RT back to ms, for legacy reasons
subplot(2,1,1); title('simulated data with DDM fit');

% compare generative and fitted params:
[k fit(1) ; B fit(2) ; Tnd fit(3)]

% Nice!

% (If your fits fail, just try again! Could be the randomization caused the
% guess to be too far off for the model to converge)

% NOTE the cool thing is that the above equations, used in the fitting
% routine, appear nowhere in the simulation code. All the simulation does
% is generate Gaussian random numbers and sum them up. The fitting code 
% uses the analytical solutions for the mean RT and pRight, and so the fact
% that they can recover the generative parameters is a kind of proof that 
% the equations are correct.

% Now it won't always be the case that there is a closed-form solution for
% the distributions or even expected values of variables in a given model.
% If that's the case, there are other methods for estimating likelihoods,
% most of which involve something akin to simulation (aka 'Monte-Carlo'
% methods, MCMC, BADS, etc). These are beyond our current scope but ask
% me about them if interested.


%%  2.4 - fit the DDM

% Now that we've proven we can recover generative params,
% let's fit the real data

clear; close all
load dots_data

% guess = [1 10 500]; % cheating a bit since I know the fitted params...
guess = [1 10 500]; % ...but turns out these don't have to be very close at all
[fit,~] = Dots_fitDDM_1D_noConf(guess,data.scoh,data.choice,round(data.RT*1000));
subplot(2,1,1); title('real data with DDM fit');

k = fit(1)
B = fit(2)
Tnd = fit(3)

% this is a remarkably good fit, except for RTs at low coh --
% something to make a new hypothesis about? or tweak the model?


% BACK TO SLIDES (if time): more on "why do we care?"




%%  3.1 - another kind of behavioral modeling: RL and bandit-style tasks

clear

% Again see Wilson & Collins 2019, and/or search on your own for 
% Rescorla-Wagner and bandit problems.

T = 100; % num trials
mu = [0.2 0.8]; % reward probabilities
alpha = 0.05; % learning rate
beta = 5; % inverse temperature

[choice, ~] = simulate_M3RescorlaWagner_v1(T, mu, alpha, beta);
choice = choice-1; % convert [1:2] to [0:1] for consistency w above

win = 10; % window width for smoothing choice probabilities
convKernel = fspecial('average', [1 win]);
chSmooth = conv(convKernel,choice);
chSmooth = chSmooth(win/2:end-win);

% plot the learning curve
figure; plot(chSmooth);
ylim([-0.1 1.1]);
xlabel('Trials');
ylabel('Choices (smoothed): 1 is ''correct''')
title('learning curve');



%%  3.1.1 - now try a non-stationary bandit

T = 1000; % num trials
mu = [0.2 0.8]; % reward probabilities
alpha = 0.05; % learning rate
beta = 5; % inverse temperature
swProb = 0.01; % probability of switching reward probs 

[choice, rew, Mu] = simulate_M3RescorlaWagner_nonStationary(T, mu, alpha, beta, swProb);
choice = choice-1; % convert [1:2] to [0:1] for consistency w above

win = 8; % window width for smoothing choice probabilities
convKernel = fspecial('average', [1 win]);
chSmooth = conv(convKernel,choice);
chSmooth = chSmooth(win/2:end-win/2);

figure;
[AX,H1,H2] = plotyy(1:T,chSmooth,1:T,Mu);
set(H1,'color','blue');
set(H2,'color','red','LineWidth',2)
ylim(AX(1),[-0.1 1.1]);
ylim(AX(2),[-0.1 1.1]);
xlabel('Trials');
ylabel(AX(1),'Choices (smoothed)')
ylabel(AX(2),'Reward probability')
 

% The general theoretical framework here is called reinforcement learning
% (RL). It is very powerful and, like the DDM and SDT, can be used to
% generate hypotheses for neural mechanisms of learning and decision
% making.

% See for example:
% Bari, B. A. et al. Stable Representations of Decision Variables
% for Flexible Behavior. Neuron 103, 922-933.e7 (2019).
 


