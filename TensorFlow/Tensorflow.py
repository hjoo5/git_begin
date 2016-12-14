## TensorFlow를 사용하기위해 라이브러리를 가져온다.
import tensorflow as tf
## MNIST 데이터를 다운로드 한다.
from tensorflow.examples.tutorials.mnist import input_data

# Dataset loading
## MNIST 데이터를 다운로드 한다.
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# Set up model
## 이미지들은 placeholder에 맞게 배치 크기별로 나뉘어진다.
x = tf.placeholder(tf.float32, [None, 784])
## 2개의 zeros모양의 variable을 만든다.
## W가 [784,10]의 형태인 이유는 784차원의 이미지 벡터를 곱해서 10차원(인코딩된 0~9)의 결과를 내기 위함.
## b는 결과에 더하기 위한 10차원.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
## 모델 구현
y = tf.nn.softmax(tf.matmul(x, W) + b)
## 정답 레이블용 플레이스 홀더
y_ = tf.placeholder(tf.float32, [None, 10])

## 손실 함수
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
## 학습 오퍼레이션
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Session
## 모든 변수 초기화
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# Learning
## 임의 1000개로 실험해본다.
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Validation
## 정답률 출력
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))