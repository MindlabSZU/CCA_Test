clc;clear;close all;
function s=signal(freq,phi)
t=linspace(0,2,500);
s=(1+sin(2*pi*freq*t+phi))/2;
end
result=zeros(500,40);
for k=1:40
    freq=8+(k-1)*0.2;
    phi=0+(k-1)*0.5*pi;
    result(:,k)=signal(freq,phi);
end