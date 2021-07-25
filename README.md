# Seq2Seq_addition
Trains a [LSTM](https://manishemirani.github.io/Long-Short-Term-Memory) model to do addition task. I did that on [pictures](https://github.com/manishemirani/Seq2Seq-picture) as well
# With tensorflow
        400/400 [==============================] - 3s 8ms/step - loss: 0.7426 - acc: 0.7251
        400/400 [==============================] - 3s 8ms/step - loss: 0.7115 - acc: 0.7375
        400/400 [==============================] - 3s 8ms/step - loss: 0.6850 - acc: 0.7442
        400/400 [==============================] - 3s 8ms/step - loss: 0.6472 - acc: 0.7573
        56+43  = 99  (correct)
        51+17  = 66  (incorrect) (correct) = 68
        61+84  = 145 (correct)
        81+64  = 145 (correct)
        400/400 [==============================] - 3s 8ms/step - loss: 0.5913 - acc: 0.7787
        400/400 [==============================] - 3s 8ms/step - loss: 0.5275 - acc: 0.8029
        400/400 [==============================] - 3s 8ms/step - loss: 0.4450 - acc: 0.8406
        400/400 [==============================] - 3s 8ms/step - loss: 0.3497 - acc: 0.8903
        26+64  = 99  (incorrect) (correct) = 90
        77+91  = 168 (correct)
        98+28  = 126 (correct)
        66+11  = 78  (incorrect) (correct) = 77
# With jax and flax
You can see the original project from [here](https://github.com/google/flax/tree/master/examples/seq2seq)        
        
        train epoch: 50, loss: 0.14621692895889282, accuracy: 89.24999833106995
        train epoch: 51, loss: 0.12401988357305527, accuracy: 91.9374942779541
        train epoch: 52, loss: 0.12541252374649048, accuracy: 90.37500023841858
        train epoch: 53, loss: 0.10723508894443512, accuracy: 93.81250143051147
        train epoch: 54, loss: 0.10625441372394562, accuracy: 92.56249666213989
        train epoch: 55, loss: 0.1099778413772583, accuracy: 91.99999570846558
