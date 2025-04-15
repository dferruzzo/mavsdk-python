% p model
clear 
close all
clc
%
w = table2array(readtable('../dados/freq_rad_s_p_rate.txt',...
    'ReadVariableNames',false,'ReadRowNames',false));
fase = table2array(readtable('../dados/fase_rad_p_rate.txt',...
    'ReadVariableNames',false,'ReadRowNames',false));
mod = table2array(readtable('../dados/modulo_dB_p_rate.txt',...
    'ReadVariableNames',false,'ReadRowNames',false));
%%
%a= 0.7827387937435848;
%T= 0.8994596936924678;
%k= 1.2497076528615696;
a= 0.6;
T= 3.0;
k= 0.9;
Ixx= 0.016;
tau= 0.0195;
%
s = tf('s');
%
ws = logspace(-1,2,100);
%
C = k*((T*s+1)/(a*T*s+1));
[mag_model_C, pha_model_C, w_model_C] = bode(C, ws);
mag_model_C_dB = 20*log10(squeeze(mag_model_C));
pha_model_C = squeeze(pha_model_C);
%
G= exp(-tau*s)*(1/(Ixx*s));
[mag_model_G, pha_model_G, w_model_G] = bode(G, ws);
mag_model_G_dB = 20*log10(squeeze(mag_model_G));
pha_model_G = squeeze(pha_model_G);
%
[mag_model_CG, pha_model_CG, w_model_CG] = bode(C*G, ws);
mag_model_CG_dB = 20*log10(squeeze(mag_model_CG));
pha_model_CG = squeeze(pha_model_CG);
%
close all;
h = figure(1);
set(h, 'WindowStyle', 'Docked');
subplot(2,1,1);
semilogx(w_model_CG, mag_model_CG_dB);   % modelo CG
hold on;
semilogx(w_model_G, mag_model_G_dB);     % modelo planta
semilogx(w_model_C, mag_model_C_dB);     % modelo compensador
semilogx(w, mod);                       % datos
semilogx(w, mod, '.');                       % datos
grid on;
legend('CG','G','C','Dados');
subplot(2,1,2);
semilogx(w_model_CG, pha_model_CG);      % modelo CG
hold on;
semilogx(w_model_G, pha_model_G);        % modelo planta
semilogx(w_model_C, pha_model_C);        % modelo compensador
semilogx(w, fase);                       % dados
semilogx(w, fase, '.');                       % datos
grid on;
legend('CG','G','C','Dados');
