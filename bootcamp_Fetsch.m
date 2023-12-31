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
% choice : animal's choice, 1=right, 0=left
% RT : reaction time, in seconds
% correct : 1=correct, 0=error

% create indexing (looping) variable
cohs = unique(data.scoh)

% preallocate 
pRight = nan(length(cohs),1);
n = nan(length(cohs),1);
for c = 1:length(cohs) % loop over signed coherence
    I = data.scoh==cohs(c); % find trials with w this particular coh
    n(c) = sum(I); % how many?
    pRight(c) = sum(I & data.choice==1) / n(c); % proportion of those trials where the choice was "right"
end
% % normally we'd want error bars, so here's the formula for standard error
% % of a proportion (derivation left as an exercise for the reader :)
% pRightSE = sqrt( (pRight.*(1-pRight)) ./ n );
% % but it's so tiny with this many trials, we can ignore it for now

figure; plot(cohs,pRight,'ko-');
title('data');
xlabel('Motion strength (%coh)');
ylabel('Proportion rightward choices');
changeAxesFontSize(gca,15,15);



%%  1.2 - descriptive model: logistic regression

% a first step is to quantify how the outcome, or dependent, variable
% (choice) varies as a function of the independent variable (coherence)

% For most outcome variables it is fine to use linear regression: y = mx+b!
% But proportions/probabilities are bounded at 0 and 1, so we use
% logistic regression: log(P/(1-P)) = mx+b
% or more commonly, logit(P) = beta0 + beta1*x + ...

% logit is a synonym for log odds,
% odds is P/(1-P) (often used in betting, eg "4 to 1 odds" = 0.8 or 80% probability)
% so logit(P) = log(P/(1-P))

% The inverse of the logit function is the logistic function: 
% P = 1 / (1 + exp(-(beta_0 + beta1*x)))
% So you often see it written this way. This is how you'd actually
% calculate the probabilities given the beta terms.

% The simplest logistic regression model for a choice function (or 
% 'psychometric' function) has two parameters (betas):
% a slope and a bias (equiv to slope and intercept for linear reg)

% The slope term is just a constant multiplied by a single independent
% variable (here, coh), but you can build a logistic model with any number
% of explanatory variables, and their (nonlinear) interactions. It is a
% very flexible and powerful tool, part of a larger family of *generalized
% linear models* (GLMs) which you will likely learn more about later.

% (On your own you can check out any number of video tutorials online)


%%  1.3 - likelihood and binomial probability

% The best-fitting model is the one that maximizes the *likelihood* of the
% data given the model parameters

% Likelihood, P(data | model), is often calculated by assuming a particular
% parametric form of the probability distribution.

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

% Let's make an initial guess for the parameters and calculate the
% single-trial likelihoods:
beta_0 = 0; % assume no bias
beta1 = 10; % random-ish guess
pR = nan(length(cohs),1);
for c = 1:length(cohs) % loop over signed coherence
    B = beta_0 + beta1*cohs(c);
    pR(c) = logistic(B);
end
figure; plot(cohs,pR,'ms-'); title('guess')
xlabel('Motion strength (%coh)');
ylabel('Proportion rightward choices');
changeAxesFontSize(gca,15,15);
% good enough!


% calculate Likelihoods
pR_model = nan(length(data.choice),1);
Lik = nan(length(data.choice),1);
for t = 1:length(data.choice) % loop over trials
    cohInd = data.scoh(t)==cohs;
    pR_model(t) = pR(cohInd);
    if data.choice(t)==1
        Lik(t) = pR_model(t);
    else
        Lik(t) = 1-pR_model(t);
    end
end
% (offline coding exercise: rewrite this without using a FOR loop)



% A snapshot of what we've done:
['    scoh       choice      pR_model     Likelihood']
[data.scoh(1:10) data.choice(1:10) pR_model(1:10) Lik(1:10)]
% In partuclar compare rows 3+4 vs. 5: in 5 the coh was positive (right)
% but monkey chose left, an error that was not well predicted by the
% model (expected pRight for that trial was high). Thus, the likelihood is
% low, 1-Pr.


% The full likelihood for the  dataset would be the product of all the
% individual likelihoods (joint probability). This calculation is made much
% simpler if we take the log-likelihood. Then we can just sum them up, 
% because log turns products into sums.


%%  1.4 - optimization

% Now that we know how to calculate the Likelihood, we could write an
% optimization function that iteratively searches over values of slope
% and bias until the Likelihood is maximal -- this is the best-fitting
% model

% Fortunately Matlab has done that for us in a very simple package, one
% that can be used for much more complicated models than just the
% 2-parameter logistic so it is worth knowing. It's called glmfit.

X = data.scoh;
y = data.choice;
    % NOTE that we are not fitting the proportions, we are fitting
    % individual trials! This is more powerful and accurate.
[Beta, ~, stats] = glmfit(X, y, 'binomial');
% (for afficionados: glmfit automatically includes a 'column of zeroes'
% in the design matrix X, to estimate the bias term)


Beta
% Beta is a two-vector with [beta0, beta1] or [bias, slope]
% Notice the bias is almost zero. The slope is in units that are not
% intuitive, because remember it's logit(P) = slope*x + bias
% but it's still true that the higher the number the steeper the slope

%% 1.4.1

% to plot our best fitting model vs. the data points, can use glmval to
% give the model-based probability of rightward given the params:

figure; set(gcf,'Color',[1 1 1],'Position',[300 500 1000 450]);
title('logistic fit')
yVals = glmval(Beta,cohs,'logit');
subplot(1,2,1); plot(cohs,pRight,'ko');
hold on; plot(cohs,yVals,'r*');
xlabel('Motion strength (%coh)');
ylabel('Proportion rightward choices');
changeAxesFontSize(gca,15,15);

% even better, when you have a model you can generate smooth curves by
% interpolating your X values:
cohsInterp = cohs(1):0.001:cohs(end);
yVals = glmval(Beta,cohsInterp,'logit');
subplot(1,2,2); plot(cohs,pRight,'ko');
hold on; plot(cohsInterp,yVals,'r-');
xlabel('Motion strength (%coh)');
ylabel('Proportion rightward choices');
changeAxesFontSize(gca,15,15);



%%  1.5 - sidebar: comparing accuracy alone can be misleading

clearvars -except cohs
close all
clc

ntrials = 50000;

% errors can occur because of bias, or because of signal-to-noise
% (sensitivity, measured by the slope)

% simulate some data with large slope but also large bias
Beta2 = [2.5 18]';
scoh = nan(ntrials,1);
choice = nan(ntrials,1);
for t = 1:ntrials
    scoh(t) = randsample(cohs,1,'true');
    P = glmval(Beta2,scoh(t),'logit');
    choice(t) = binornd(1,P);
end
correct = zeros(ntrials,1);
correct(scoh>0 & choice==1) = 1;
correct(scoh<0 & choice==0) = 1;
correct(scoh==0) = randn>0;
pctCorr1 = sum(correct)/ntrials

% plot
pRight1 = nan(length(cohs),1);
n = nan(length(cohs),1);
for c = 1:length(cohs)
    I = scoh==cohs(c);
    n(c) = sum(I);
    pRight1(c) = sum(I & choice==1) / n(c);
end
figure(5); clf;
plot(cohs,pRight1(:,1),'ko-');
xlabel('Motion strength (%coh)');
ylabel('Proportion rightward choices');
changeAxesFontSize(gca,15,15);


%% 1.5.1

% now simulate unbiased choices but with lower slope (less sensitivity)
Beta2 = [0 8]';
scoh = nan(ntrials,1);
choice = nan(ntrials,1);
for t = 1:ntrials
    scoh(t) = randsample(cohs,1,'true');
    P = glmval(Beta2,scoh(t),'logit');
    choice(t) = binornd(1,P);
end
correct = zeros(ntrials,1);
correct(scoh>0 & choice==1) = 1;
correct(scoh<0 & choice==0) = 1;
correct(scoh==0) = randn>0;
pctCorr2 = sum(correct)/ntrials

% plot
pRight2 = nan(length(cohs),1);
n = nan(length(cohs),1);
for c = 1:length(cohs)
    I = scoh==cohs(c); 
    n(c) = sum(I);
    pRight2(c) = sum(I & choice==1) / n(c);
end
figure(5); hold on; plot(cohs,pRight2(:,1),'bs-');
changeAxesFontSize(gca,15,15);


% accuracy (percent correct) will be about the same, but clearly this
% does not tell the whole story



%%  2.1 - signal-detection theory (SDT)

clear scoh
ntrials = 50000;

% A logistic model is nice but merely descriptive. It describes average
% behavior over ensembles of trials (a given set of independent variables),
% but we want to know how the brain makes a decision on a single trial.
% For this we need a mechanistic or 'process' model.

% (Sometimes this is described is a 'generative' model, but that term is
% used differently in some areas of research)

% In SDT, a stimulus is said to give rise to a noisy 'observation' or
% measurement (in the brain). You can think of it as something like the
% number of spikes in a population of neurons over a given time interval.
% (This makes most sense for detecting a faint signal embedded in noise;
% in the case of discriminating or categorizing stimuli (eg Left/Right)
% it's better to think of the observation as the difference in spike counts 
% between two populations, one selective for Right and the other for Left)

% The observation is thus a random variable with some distribution. Let's
% assume Gaussian, with mean proportional to the stimulus strength (coh).
% For simplicity we'll assume the standard deviation is a constant

k = 8; % proportionality constant converting coh to the mean of the signal distribution
sigma = 1; % s.d. of the signal distribution 
scoh = nan(ntrials,1);
obs = nan(ntrials,1);
choice = nan(ntrials,1);
for t = 1:ntrials
    scoh(t) = randsample(cohs,1,'true');
    mu = k*scoh(t);
    obs(t) = normrnd(mu, sigma);
    
    % SDT also involves a criterion:
    % if the obs is greater than the criterion you make a certain choice.
    % because coherence is signed, the obvious criterion is simply zero; if 
    % greater than zero, choose Right, else choose Left
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
figure(6); clf;
plot(cohs,pRight,'go-');
xlabel('Motion strength (%coh)');
ylabel('Proportion rightward choices');
title('simulated data: SDT');
changeAxesFontSize(gca,15,15);


% For today we'll stop at stimulation, but this model could be fit to data
% and used to generate hypotheses about the neural basis of signal, noise,
% and criterion. 

% So although it still seems rather abstract, SDT was among the first
% generative/process models for perception that could be  translated into a
% so-called 'linking hypothesis'; this was arguably the genesis of
% (sensory) systems neuroscience

% (BACK TO SLIDES)





%%  2.2 - what if we want to explain more than just the choice?

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

figure; set(gcf,'Color',[1 1 1],'Position',[300 500 450 600]);
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

% Let's simulate it.

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

% Of course we want to do more than just simulate data that look reasonable.
% We want to actually *fit* our process model to the data.  Why?
% (1) to verify that the model is a reasonable one
% (2) to compare goodness of fit across different candidate models
% (3) to compare fitted parameters across conditions/subjects and thereby
%     generate hypotheses as to what might differ

% But before we can trust our model fits, we should check that
% our experiment and fitting code are capable, in principle, of correctly
% arriving at the 'ground truth'. To do this we can generate fake data
% by SIMULATING the model (in which case obviously we know the ground truth
% aka the generative parameters), then FIT the fake data to try and recover
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

guess = [k B Tnd]; % assumes sigma=1
[fit,~] = Dots_fitDDM_1D_noConf(guess,data.scoh,data.choice,round(data.RT*1000)); % convert RT back to ms, for legacy reasons
subplot(2,1,1); title('simulated data with DDM fit');

% compare generative and fitted params:
[k fit(1) ; B fit(2) ; Tnd fit(3)]

% Nice!

% (NOTE the cool thing is that the above equations, used in the fitting
% routine, appear nowhere in the simulation code. All the simulation does
% is generate Gaussian random numbers and sum them up. The fitting code 
% uses the analytical solutions for the mean RT and pRight, and so the fact
% that they can recover the generative parameters is a kind of proof that 
% the equations are correct.)

% Now it won't always be the case that there is a closed-form solution for
% the distributions or even expected values of variables in a given model.
% If that's the case, there are other methods for estimating likelihoods,
% most of which involve something akin to simulation (aka 'Monte-Carlo'
% methods, MCMC, BADS, etc). These are out of our current scope but will
% send some refs.


%%  2.4 - now that we've proven we can recover generative params, let's fit the real data

clear; close all
load dots_data

% guess = [1 10 500]; % cheating a bit since I know the fitted params...
guess = [1 10 500]; % ...but turns out these don't have to be very close at all
[fit,~] = Dots_fitDDM_1D_noConf(guess,data.scoh,data.choice,round(data.RT*1000));
subplot(2,1,1); title('real data with DDM fit');

k = fit(1)
B = fit(2)
Tnd = fit(3)

% seems pretty good except for RTs at low coh -- something to make a new
% hypothesis about? or tweak the model to better explain it?


% BACK TO SLIDES (if time): more on "why do we care?"




%%  3.1 (if time) - another kind of behavioral modeling: RL and bandit-type tasks

clear

% Again see Wilson & Collins 2019, and/or search on your own for 
% Rescorla-Wagner and bandit problems.

T = 100; % num trials
mu = [0.2 0.8]; % reward probabilities
alpha = 0.05; % learning rate
beta = 5; % inverse temperature

[choice, ~] = simulate_M3RescorlaWagner_v1(T, mu, alpha, beta);

win = 10; % window width for smoothing choice probabilities
convKernel = fspecial('average', [1 win]);
chSmooth = conv(convKernel,choice);
chSmooth = chSmooth(win/2:end-win);

% plot the learning curve
figure; plot(chSmooth);
ylim([0.9 2.1]);
xlabel('Trials');
ylabel('Choices (smoothed): 2 is ''correct''')
title('learning curve');



%%  3.1.1 - now try non-stationary bandit

T = 1000; % num trials
mu = [0.3 0.7]; % reward probabilities
alpha = 0.1; % learning rate
beta = 5; % inverse temperature
swProb = 0.01; % probability of switching Mu's 

[choice, rew, Mu] = simulate_M3RescorlaWagner_nonStationary(T, mu, alpha, beta, swProb);

win = 10; % window width for smoothing choice probabilities
convKernel = fspecial('average', [1 win]);
chSmooth = conv(convKernel,choice);
chSmooth = chSmooth(win/2:end-win/2);

figure;
[AX,H1,H2] = plotyy(1:T,chSmooth,1:T,Mu);
set(H1,'color','blue');
set(H2,'color','green','LineWidth',2)
ylim([0.9 2.1]);
xlabel('Trials');
ylabel('Choices (smoothed)')
 
% see also:
% Bari, B. A. et al. Stable Representations of Decision Variables
% for Flexible Behavior. Neuron 103, 922-933.e7 (2019).
  


%% more cool stuff to play with on your own:

% https://pubmed.ncbi.nlm.nih.gov/33412101/
% https://github.com/nicholas-roy/PsyTrack










