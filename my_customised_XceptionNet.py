#** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

def my_customisable_XceptionNet(input_shape, number_of_inner_blocks=8, short_network=True, number_of_output_units=4):
    X_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3),
                      strides=(2, 2),
                      use_bias=False,
                      name='block1_conv1')(X_input)
    x = layers.BatchNormalization(name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv1')(x)
    x = layers.BatchNormalization(name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv2')(x)
    x = layers.BatchNormalization(name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv1')(x)
    x = layers.BatchNormalization(name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv2')(x)
    x = layers.BatchNormalization(name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(728, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(x)
    x = layers.BatchNormalization(name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv2')(x)
    x = layers.BatchNormalization(name='block4_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(number_of_inner_blocks):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv3')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    if (short_network):
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block13_sepconv1')(x)
        x = layers.BatchNormalization(name='block13_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block13_sepconv2_act')(x)
        x = layers.GlobalAveragePooling2D(name='avg_pool1')(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(number_of_output_units, activation='relu', name='predictions')(x)


    else:
        residual = layers.Conv2D(1024, (1, 1), strides=(2, 2),
                                 padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.Activation('relu', name='block13_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block13_sepconv1')(x)
        x = layers.BatchNormalization(name='block13_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block13_sepconv2_act')(x)
        x = layers.SeparableConv2D(1024, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block13_sepconv2')(x)
        x = layers.BatchNormalization(name='block13_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3),
                                strides=(2, 2),
                                padding='same',
                                name='block13_pool')(x)
        x = layers.add([x, residual])

        x = layers.SeparableConv2D(1536, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block14_sepconv1')(x)
        x = layers.BatchNormalization(name='block14_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block14_sepconv1_act')(x)

        x = layers.SeparableConv2D(2048, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block14_sepconv2')(x)
        x = layers.BatchNormalization(name='block14_sepconv2_bn')(x)
        x = layers.Activation('relu', name='block14_sepconv2_act')(x)
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(number_of_output_units, activation='relu', name='predictions')(x)

    model = models.Model(inputs=X_input, outputs=x, name='Xception')

    return model













