# Definition of vectors in MATLAB.
v = [3, 2, 5]
u = [2, -1, 0]

# Operations: Addition, Subtraction, Multiplication, and Division with matrices.
a = [ 10 12 23 ; 14 8 6; 27 8 9];
b = 2;
c = a + b
d = a - b
e = a * b
f = a / b

# Adding two vectors.
v = [3, 2, 5]
u = [2, -1, 0]
v + u

# Norms: Various types of norms such as L2 norm, infinity norm, Frobenius norm, etc.
A = [ 10 12 23 ; 14 8 6; 27 8 9]
norm(A, 1) 
norm(A,inf)
norm(A, 2) 
norm(A,'fro')

# Angles between vectors and axes.
atan2(6, -2)*180/pi

# More examples of norms.

### Create a vector and calculate the magnitude.
v = [1 -2 3];
n = norm(v)

### Calculate the 1-norm of a vector, which is the sum of the element magnitudes.
v = [-2 3 -1];
n = norm(v, 1)

### Calculate the distance between 2 points
a = [0 3];
b = [-2 1];

d = norm(b-a)

### Calculate the 2-norm which is the largest singular value
X = [2 0 1; -1 1 0; -3 3 0];
n = norm(X)

# Angle between a vector and axis x.
atan2(6, -2)*180/pi

# Calculating the angle between two vectors.
u = [1 1 0]
v = [0 1 0]
CosTheta = max(min(dot(u,v)/(norm(u)*norm(v)),1),-1);
ThetaInDegrees = real(acosd(CosTheta))

# Row and column vectors.
### row vector
r = [7 8 9 10 11]
### column vector
c = [7;  8;  9;  10; 11]

# Accessing a specific element of a vector.
v = [ 1; 2; 3; 4; 5; 6];
v(3)

# Zero vector and unit vector.

### row and column vector of zeros 
n = 8;
zeros(1,n)
zeros(n,1)
## row and column vector of ones 
ones(1,n)
ones(n,1)

# Dot product of two vectors.
a = [-2, 6]
b = [6, 3]
dot (a, b)

# Calculating the sum of all elements of a vector.
x = [8 5 6]
y = [5; 7; 9]
sum(x)
sum(y)

# Accessing a specific element of a matrix.
a = [ 1 2 3 4 5; 2 3 4 5 6; 3 4 5 6 7; 4 5 6 7 8];
a(2,5)

# Accessing a specific column of a matrix.
A = [ 1 2 3 4 5; 2 3 4 5 6; 3 4 5 6 7; 4 5 6 7 8];
v = A(:,4)

# Accessing a specific row of a matrix.
A = [ 1 2 3 4 5; 2 3 4 5 6; 3 4 5 6 7; 4 5 6 7 8];
v = A(3,:)

# Creating submatrices from matrices.
a = [ 1 2 3 4 5; 2 3 4 5 6; 3 4 5 6 7; 4 5 6 7 8];
sa = a(2:3, 2:4)

# Generating zero and unit matrices.
zeros(5)
ones(4,3)

# Generating identity matrices.
eye(4)

# The magic matrix command.
magic(4)

# Concatenating matrices horizontally and vertically.
A = [ 1 2 3; 4 5 6; 7 8 9];
B = [ 0 0 0; 1 1 1; 2 2 2];
[A, B]
[A;B]

# Augmented matrices.
A = [ 1 2 3; 4 5 6; 7 8 9];
b = [ 1 ;-1 ;0];
[A, b]

# Dimensions of a matrix.
A = [ 1 2 3 4 5; 2 3 4 5 6; 3 4 5 6 7; 4 5 6 7 8];
size(A)

# Generating random matrices.

### 3 * 5
rand(3, 5)

### Specified range
n = 5
a = 10; b = 50;
x = a + (b-a) * rand(n)

### r = 5*5
r = rand(5)

### r = 10*1
r = -5 + (5+5)*rand(10,1)

### r = 1*5 and numbers within the specified range
>>r = randi([10 50],1,5)

# Calculating the sum of all elements in a matrix.
A = [ 1 2 3; 4 5 6; 7 8 9];
sum(A(:))

# Matrix addition and subtraction.
a = [ 1 2 3 ; 4 5 6; 7 8 9];
b = [ 7 5 6 ; 2 0 8; 5 7 1];
c = a + b
d = a - b

# Point-wise multiplication of two matrices.
A = [ 1 2 3 ; 4 5 6; 7 8 9];
B = [ 7 5 6 ; 2 0 8; 5 7 1];
A.*B

# Matrix multiplication of two matrices.
a = [ 1 2 3; 2 3 4; 1 2 5]
b = [ 2 1 3 ; 5 0 -2; 2 3 -1]
prod = a * b

# Raising a matrix to a power.
a = [ 1 2 3; 2 3 4; 1 2 5]
a ^ 2
a ^ 3

# Raising elements of a matrix to a power.
a = [ 1 2 3; 2 3 4; 1 2 5];
a.^2

# Matrix operations / and \.
a = [ 1 2 3 ; 4 5 6; 7 8 9];
b = [ 7 5 6 ; 2 0 8; 5 7 1];
c = a / b
d = a \ b

# Calculating the determinant.
a = [ 1 2 3; 2 3 4; 1 2 5]
det(a)

# Computing eigenvalues of a matrix.
A = [0 1 2; 
    1 0 -1; 
    2 -1 0]
e = eig(A)
max(abs(eig(A)))

# Obtaining the radius of a matrix and displaying it as a vector.
A = [ 1 2 3; 2 3 4; 1 2 5]
diag(A)

# Calculating the trace of a matrix.
A = [ 1 2 3; 2 3 4; 1 2 5]
trace(A)

# Calculating the rank of a matrix.
A = [ 1 2 3; 2 3 4; 1 2 5]
rank(A)

# Inverse of a matrix.
A = [1, 2; 3, 4]
inv(A) 
inv(A)*A

# Pseudo Inverse of a matrix.
A = magic(8) 
A = A(:,1:6)

# Calculating upper and lower triangular matrices.
A = [1 2 3; 4 5 6; 7 8 9]
upperDiagram = triu(A,1)
lowerDiagram = tril(A,-1)

# All possible permutations of a matrix.
v = [2 4 6];
P = perms(v)

# Checking if a matrix is Orthogonal or not.
a = [0,1 ; -1,0];
b = [2,3 ; 4,-3]

% Function to check if a matrix is orthogonal
function isOrthogonal = checkOrthogonal(a)
    [m, n] = size(a);
    if m ~= n
        isOrthogonal = false;
       return;
    end

    % Find transpose
    trans = a';

    % Find product of a and its transpose
    prod = a * trans;

    % Check if product is identity matrix
    identity = eye(n);
    isOrthogonal = isequal(prod, identity);
end
if checkOrthogonal(a)
    disp('Yes');
else
    disp('No');
end
if checkOrthogonal(b)
    disp('Yes');
else
    disp('No');
end
# Eigenvectors of matrices.
رA = [10 -6 2; 
    -6 7 -4; 
     2 -4 3];   
[V,D,W] = eig(A); 
disp("Right eigenvectors :") 
disp(V); 

# Check idempotent property of matrix
mat = [2, -2, -4; -1, 3, 4; 1, -2, -3];
mat2 = [2, -1, -3; 1, 3, -4; 4, 2, -3];

% Function for matrix multiplication
function res = multiply(mat)
    N = size(mat, 1);
    res = zeros(N, N);
    for i = 1:N
        for j = 1:N
            for k = 1:N
                res(i, j) = res(i, j) + mat(i, k) * mat(k, j);
            end
        end
    end
end

% Function to check idempotent property of matrix
function isIdempotent = checkIdempotent(mat)
    N = size(mat, 1);
    res = multiply(mat);
    isIdempotent = isequal(mat, res);
end
% checkIdempotent function call
if checkIdempotent(mat)
    disp('Idempotent Matrix');
else
    disp('Not Idempotent Matrix');
end
if checkIdempotent(mat2)
    disp('Idempotent Matrix');
else
    disp('Not Idempotent Matrix');
end
# Check Nilpotent property of matrix
mat = [5, -3, 2; 15, -9, 6; 10, -6, 4]
flag = 1;

N = size(mat, 1);
res = zeros(N, N);

for i = 1:N
    for j = 1:N
        for k = 1:N
            res(i, j) = res(i, j) + mat(i, k) * mat(k, j);
        end
    end
end
for i = 1 : 3
  for j = 1 : 3
    if res(i,j) != 0
       disp('Not Nilpotent Matrix')
       flag = 0;
    endif
  endfor
end

if flag == 1
    disp('Nilpotent Matrix');
end

# Sorting elements of an array.
v = [ 23 45 12 9 5 0 19 17]  % horizontal vector
sort(v)                      % sorting v

# Sorting elements of a matrix (row-wise and column-wise).
m = [2 6 4; 5 3 9; 2 0 1]    % two dimensional array
sort(m, 1)                   % sorting m along the row
sort(m, 2)                   % sorting m along the column

# Solving linear equations with three different commands.
A = [3, 4, 5;2, -3, 7;1, -6, 1]
b = [2; -1; 3] 
x = A \ b

x = inv(A) * b

X = linsolve(A,b)

# Identifying stable or unstable states of a problem.
A = [2, 2; 2, 2.01];
b = [1, -1];

delta_A = [-0.001, 0; 0, 0];
delta_B = [0.0001; 0.0001];

d = 4;
k = 2;
result = 10 ^ (d - k - 1)

k_1 = norm(A,1) * norm(inv(A),1)

if k_1 > 10
  disp("ill-conditioned");
else
  disp("well-conditioned");
end
# Projection of 2 vectors.
v = [1,1]
u = [2,0]

proj = (dot(u,v)/(norm(u,2)^2))*u

# Solving specific systems of equations.
[X,e] = polyeig(6,-5,1)

# Calculating individual vector values.
A = [-5, 1; 8, 0; 4, -7];
S = svds(A)

# Calculating individual vectors.
A = [-1,0,2;1,2,-2]

[U,S,V] = svd(A)

first = V(:, 1)
second = V(:, 2)
third = V(:, 3)

# Solving specific systems of equations.
[X,e] = polyeig(6,-5,1)

# Implementing the Gram-Schmidt algorithm.
A = [1, 3, 2; -1, 0, 6; 3, 8, 5]

[m, n] = size(A)
Q = zeros(m, n)
R = zeros(n, n)

for j=1 : n
  v = A(:,j);
  for i=1 : j-1
    R(i, j) = Q(:,j)'*A(:,j);
    v = v - R(i,j) * Q(:,i);
  endfor
  R(j, j) = norm(v)
  Q(:,j) = v/R(j,j)
end
Q
R
Q * R

# Computing QR decomposition.
A = magic(5)
[Q, R] = qr(A)

# Computing SVD decomposition.
A = [1 2; 3 4; 5 6]
[U, S, V] = svd(A)

# Plotting the linear equations of a two-dimensional system.
x = linspace(1, 3, 100)
y = 4 - 3*x
plot(x,y)
hold on
y = (x - 6)/2
plot(x, y)

x = linspace(1, 3, 100)
y1 = 4 - 3*x
y2 = 8 - 3*x
figure
plot(x,y1,x,y2)

# Creating and labeling each line graph.
#    - How to create and label names of plots.
#    - How to display plots in a grid layout.
#    - Changing the color of plotted lines

x = linspace(1, 3, 100)
y1 = 4 - 3*x
y2 = (x - 6)/2
figure
plot(x,y1,'--',x,y2,':'), legend('First', 'Second'),title('Title'), grid on

x = linspace(1, 3, 100)
y1 = 4 - 3*x
y2 = (x - 6)/2
figure
plot(x,y1,'b',x,y2,'m'), legend('First', 'Second'), xlabel('x'), ylabel('y'),title('Title'), grid on

# Plotting the linear equations of a three-dimensional system in four scenarios.
# single point
[x,y] = meshgrid(-5:0.5:5)
z = ( 4 + 2*x - 6*y)/(-2)
surf(x,y,z)
hold on
z = (8 - 4*x + y)/2
surf(x,y,z)
hold on
z = (-6 + 3*x + 3*y)
surf(x,y,z)

# parallel
[x,y] = meshgrid(-5:0.5:5)
z = 10 - x - y
surf(x,y,z)
hold on
z = (60 - 2*x - 2*y)/2
surf(x,y,z)
hold on
z = (50 - 3*x - 3*y)/3
surf(x,y,z)

# Coincident planes
[x,y] = meshgrid(-5:0.5:5)
z = 10 - x - y
surf(x,y,z)
hold on
z = (20 - 2*x - 2*y)/2
surf(x,y,z)
hold on
z = (30 - 3*x - 3*y)/3
surf(x,y,z)

# Two planes intersecting with the third plane parallel to them
[x,y] = meshgrid(-5:0.5:5)
z = (-2*x-3*y-5)/4
surf(x,y,z)
hold on
z = (-2*x-3*y-8)/4
surf(x,y,z)
hold on
z = (-5*x-2*y-1)/3
surf(x,y,z)

# Null Space.
A = [ 1 2 3; 4 5 6; 7 8 9]
N = null(A)
A * N
norm(A * N)
A = ones(3)
x1 = null(A)
A * x1
norm(A * x1)

# Rank Space.
A = [1 0 1; -1 -2 0; 0 1 -1]
r = rank(A)
Q = orth(A)

# PLU decomposition.
A = [4, 3, 2; 6, 3, 1; 0, 5, 6]
[L, U, P] = lu(A)

# Cholesky decomposition.
A = [8 2 3; 3 5 6; 7 5 9]
R = chol(A)
# LL^T
R' * R

# Identifying if a matrix is positive definite using two methods.
# Try Choleski decomposition
A = [1 -1 0; -1 5 0; 0 0 7]
try chol(A)
    disp( 'Matrix is symmetric positive definite.' )
  catch ME
    disp( 'Matrix is not symmetric positive definite.' )
  end

A = [5 2 6; 0 5 9; -9 -8 7]
try chol(A)
    disp( 'Matrix is symmetric positive definite.' )
  catch ME
    disp( 'Matrix is not symmetric positive definite.' )
  end

# Check eigenvalues
# using d>=0 you can check whether a matrix is a symmetric positive semidefinite matrix
A = [1 -1 0; -1 5 0; 0 0 7]
tf = issymmetric(A)
d = eig(A)
isposdef = all(d > 0)
B = [5 2 6; 0 5 9; -9 -8 7]
tf = issymmetric(B)
d = eig(B)
isposdef = all(d > 0)

# Loops in MATLAB.
# for loop
x = ones(1, 10)
for n=2:6
  x(n) = 2 * x(n - 1)
end
# while loop
n = 1;
max = 1;
while max < 100
  n = n + 1
  max = max * n;
end
# Nested Loops
A = zeros(2, 3)
for  m = 1:2
  for n = 1:3
    A(m, n) = 2 * (m + n)
  end
end

# 70. Implementing iterative methods for solving linear equations using matrix and point-based methods.
A = [10 1 -1; 3 -5 4; 1 -3 10];
b = [9; 5; 25];
n = length(A);
L = tril(A, -1);
U = triu(A, 1);
D = diag(diag(A));
x0 = zeros(n,1);
MJ = -D\(L+U);
cJ = D \ b;
k = 0;

while k <= 5
  k = k + 1
  X = MJ * x0 + cJ
  x0 = X;
end

e = A \ b


A = [10 1 -1; 3 -5 4; 1 -3 10];
b = [9; 5; 25];
a = A;
n = length(A);
x0 = zeros(n, 1);
Iteration = 10;
for i = 1:n
  x(i) = ((b(i) - a(i,[1:i - 1, i + 1:n]) * x0([1: i - 1, i + 1:n]))/a(i,i))
  end
x1 = x';
for k=1:Iteration
  for i=1:n
    xx(i) = ((b(i) - a(i, [1:i-1,i+1:n]) * x1([1:i-1,i+1:n]))/a(i,i))
  end
  x1=xx';
end           
