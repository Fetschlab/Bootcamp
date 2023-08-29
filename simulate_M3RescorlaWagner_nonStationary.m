function [a, r, Mu] = simulate_M3RescorlaWagner_nonStationary(T, mu, alpha, beta, switchProb)

Q = [0.5 0.5];
Mu = nan(1,T);

for t = 1:T
        
    % compute choice probabilities
    p = exp(beta*Q) / sum(exp(beta*Q));
    
    % make choice according to choice probababilities
    a(t) = choose(p);
    
    % generate reward based on choice
    r(t) = rand < mu(a(t));
    
    % update values
    delta = r(t) - Q(a(t));
    Q(a(t)) = Q(a(t)) + alpha * delta;
    
    % track the changing mu's
    Mu(t) = mu(2);
    
    % with some probability, switch the mu's
    if rand < switchProb
        temp=mu(1);
        mu(1)=mu(2);
        mu(2)=temp;
    end
   
end

