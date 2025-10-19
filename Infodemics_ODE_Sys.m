%% System of Differential Equations and Time Trajectories
% This script solves and plots the trajectories of a 5D system with given parameters

clear; close all; clc;

%% Parameters
gam     = 0.07;
chi     = 0.048;
chi_bar = 0.0048;
chi_hat = 0.37;
epsilon = 0.33;
mu      = 0.066; % Fig 2 cases: 0.10 (Default), 1e-6 (DFE), 0.25 (Co-Existence), 0.066 (Limit Cycle)
r       = 1.635;
s       = 1;

%% Initial Conditions
sg0  = 0.15;
sb0  = 0.45;
ib0  = 0.08;
v0   = 0.15;
phi0 = 1e-3;
y0   = [sg0 sb0 ib0 v0 phi0];

%% Function for the system of ODEs
f = @(t,y) systemODE(t,y,gam,chi,chi_bar,chi_hat,epsilon,mu,r,s);

%% Time horizons
tshort = [0 150];
tlong  = [0 15000];

%% Solve system (short and long)
opts = odeset('RelTol',1e-8,'AbsTol',1e-10);
[tS, yS] = ode45(f, tshort, y0, opts);
[tL, yL] = ode45(f, tlong,  y0, opts);

%% Compute derived quantities (short)
[igS, IS, GS, BS, ReS] = derivedQuantities(yS,gam,chi_bar,chi_hat,epsilon);

%% Compute derived quantities (long)
[igL, IL, GL, BL, ReL] = derivedQuantities(yL,gam,chi_bar,chi_hat,epsilon);

%% ---- Plot Time Trajectories (Short Time Scale) ----
figure('Name','Short Time Scale','Position',[100 100 900 700]);
plot(tS, yS(:,1), 'b', 'LineWidth',1.5); hold on;  % sg (blue)
plot(tS, yS(:,2), 'r', 'LineWidth',1.5);           % sb (red)
plot(tS, yS(:,3), 'r--', 'LineWidth',1.5);         % ib (red dashed)
plot(tS, yS(:,4), 'g', 'LineWidth',1.5);           % v (green)
plot(tS, yS(:,5), 'k', 'LineWidth',1.5);           % phi (black)
xlabel('Time'); ylabel('State Variables');
title('State Variables over Short Time Scale');
legend('off') % Legend off for short time scales
grid on;

%% ---- Plot Time Trajectories (Long Time Scale) ----
figure('Name','Long Time Scale','Position',[100 100 900 700]);
plot(tL, yL(:,1), 'b', 'LineWidth',1.5); hold on;
plot(tL, yL(:,2), 'r', 'LineWidth',1.5);
plot(tL, yL(:,3), 'r--', 'LineWidth',1.5);
plot(tL, yL(:,4), 'g', 'LineWidth',1.5);
plot(tL, yL(:,5), 'k', 'LineWidth',1.5);
xlabel('Time'); ylabel('State Variables');
title('State Variables over Long Time Scale');
legend('S_G','S_B','I_B','V','\phi','Location','best');
grid on;

%% ---- Plot Auxiliary Functions (Short) ----
figure('Name','Auxiliary Functions (Short)','Position',[100 100 900 700]);
plot(tS, IS, 'k', 'LineWidth',1.5); hold on;   % I = ig + ib
plot(tS, GS, 'b', 'LineWidth',1.5);            % G = sg + ig
plot(tS, BS, 'r', 'LineWidth',1.5);            % B = sb + ib
plot(tS, yS(:,4), 'g', 'LineWidth',1.5);       % v
xlabel('Time'); ylabel('Auxiliary Variables');
title('Auxiliary Functions (Short Time Scale)');
legend('off') % Legend off for short time scales
grid on;

%% ---- Plot Auxiliary Functions (Long) ----
figure('Name','Auxiliary Functions (Long)','Position',[100 100 900 700]);
plot(tL, IL, 'k', 'LineWidth',1.5); hold on;
plot(tL, GL, 'b', 'LineWidth',1.5);
plot(tL, BL, 'r', 'LineWidth',1.5);
plot(tL, yL(:,4), 'g', 'LineWidth',1.5);
xlabel('Time'); ylabel('Auxiliary Variables');
title('Auxiliary Functions (Long Time Scale)');
legend('I = I_G + I_B','G = S_G + I_G','B = S_B + I_B','V','Location','best');
grid on;

%% ---- Plot Effective Reproductive Number (Short) ----
figure('Name','Effective Reproductive Number','Position',[100 100 900 700]);
plot(tS, ReS, 'b', 'LineWidth',1.5);
xlabel('Time'); ylabel('R_e(t)');
title('Effective Reproductive Number (Short Time Scale)');
legend('off') % Legend off for short time scales
grid on;

figure('Name','Effective Reproductive Number (Long)','Position',[100 100 900 700]);
plot(tL, ReL, 'b', 'LineWidth',1.5);
xlabel('Time'); ylabel('R_e(t)');
title('Effective Reproductive Number (Long Time Scale)');
legend('off') 
grid on;


%% ------------------------------------------------------------------------
% Supporting Functions
% ------------------------------------------------------------------------
function dydt = systemODE(~,y,gam,chi,chi_bar,chi_hat,epsilon,mu,r,s)
    sg  = y(1);
    sb  = y(2);
    ib  = y(3);
    v   = y(4);
    phi = y(5);

    ig = 1 - (sg + sb + ib + v);

    sgp  = gam*ig - phi*sg - chi*ib*sg - mu*sg*(sb + ib);
    sbp  = gam*ib + mu*sg*(sb + ib) - chi_hat*ib*sb;
    ibp  = ib*(-gam + chi_hat*sb + chi_bar*v - epsilon*(sg + ig));
    vp   = phi*sg - chi_bar*ib*v;
    phip = s*phi*(1 - phi)*(ig + ib - r*v);

    dydt = [sgp; sbp; ibp; vp; phip];
end

function [ig, I, G, B, Re] = derivedQuantities(y,gam,chi_bar,chi_hat,epsilon)
    sg = y(:,1); sb = y(:,2); ib = y(:,3); v = y(:,4);
    ig = 1 - (sg + sb + ib + v);
    I  = ig + ib;
    G  = sg + ig;
    B  = sb + ib;
    Re = (chi_hat .* sb + chi_bar .* v) ./ (gam + epsilon .* (sg + ig));
end
