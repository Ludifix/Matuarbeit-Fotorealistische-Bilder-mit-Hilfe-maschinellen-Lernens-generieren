import tensorflow as tf
import numpy as np
# Datensatz
import tensorflow_datasets as tfds
# Schichten für die Netzwerke
from tensorflow.keras.layers import Dense, Conv2DTranspose, BatchNormalization, LeakyReLU, Reshape
from tensorflow.keras.layers import Conv2D, ReLU, Flatten, Dense

import os

# HYPERPARAMETER
DATENSATZ_NAME = 'celeb_a'
BILD_GRÖSSE = (64, 64)
EPOCHEN = 50
TRAININGS_SCHRITTE_PRO_EPOCHE = 1024
BATCH_GRÖSSE = 64
GENERATOR_LERNRATE = 0.0005
DISKRIMINATOR_LERNRATE = 0.0004			

SEED = tf.random.normal([3, 100])


# ----------------------------------------- Datensatz laden und vorverarbeiten -----------------------------------------


def datsensatz_laden_und_vorverarbeiten(name, batch_grösse, bild_grösse):
	datensatz = tfds.load(name=name, split='train', download=False)

	def vorverarbeiten(example, bild_grösse=bild_grösse):
		# Diese Funktion wird auf jedes Bild des Datensatzes angewendet, wenn es geladen wird.
		bild = example['image']
		bild = tf.image.resize(bild, bild_grösse) # Grösse des Bildes wird geändert.
		bild = (bild / 127.5) - 1 # Bilder werden von [0;256] auf [-1;1] skaliert.
		return bild

	datensatz = datensatz.map(vorverarbeiten, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	datensatz = datensatz.batch(batch_size=batch_grösse)
	return datensatz


# -------------------------------------------  Generator und Diskriminator  --------------------------------------------


class Generator(tf.keras.Model):
  
	def __init__(self, eingabevektor_länge=100):
		super().__init__(name='generator')
		
		# Eingabeschicht, Form: (100,) -> (4, 4, 256)
		self.input_layer = Dense(units=4*4*256, input_shape=(eingabevektor_länge,))
		self.reshape_1 = Reshape((4, 4, 256))

		# Block 1, Form: (4, 4, 256) -> (8, 8, 128)
		self.transposed_conv2d_1 = Conv2DTranspose(filters=128,kernel_size=5, strides=(2,2),padding='same')
		self.batch_norm_1 = BatchNormalization()
		self.relu_1 = ReLU()

		# Block 2, Form: (8, 8, 128) -> (16, 16, 64)
		self.transposed_conv2d_2 = Conv2DTranspose(filters=64, kernel_size=5, strides=(2,2), padding='same')
		self.batch_norm_2 = BatchNormalization()
		self.relu_2 = ReLU()

		# Block 3, Form: (16, 16, 64) -> (32, 32, 32)
		self.transposed_conv2d_3 = Conv2DTranspose(filters=32, kernel_size=5, strides=(2,2), padding='same')
		self.batch_norm_3 = BatchNormalization()
		self.relu_3 = ReLU()

		# Ausgabeschicht, Form: (32, 32, 32) -> (64, 64, 3)
		self.output_layer = Conv2DTranspose(filters=3, kernel_size=5, strides=(2,2), padding='same', activation='tanh')
  
	def call(self, input_tensor):
		x = self.input_layer(input_tensor)
		x = self.reshape_1(x)
		# (4, 4, 256)
		x = self.transposed_conv2d_1(x) 
		x = self.batch_norm_1(x)
		x = self.relu_1(x) 
		# (8, 8, 128)    
		x = self.transposed_conv2d_2(x) 
		x = self.batch_norm_2(x)
		x = self.relu_2(x) 
		# (16, 16, 64)
		x = self.transposed_conv2d_3(x)
		x = self.batch_norm_3(x)
		x = self.relu_3(x)
		# (32, 32, 32)
		return self.output_layer(x)  # (64, 64, 3)
	
	def generiere_eingabevektor(self, batch_grösse, länge):
		return tf.random.normal(shape=[batch_grösse, länge])


class Diskriminator(tf.keras.Model):
	def __init__(self):
		super().__init__(name="diskriminator")

		# Eingabeschicht, Form: (64, 64, 3) -> (32, 32, 32)
		self.input_layer = Conv2D(filters=32, kernel_size=5, strides=(2, 2), padding='same', input_shape=(64, 64, 3))
		self.leaky_0 = LeakyReLU(alpha=0.2)

		# Block 1, Form: (32, 32, 32) -> (16, 16, 64)
		self.conv2d_1 = Conv2D(filters=64, kernel_size=5, strides=(2, 2), padding='same')
		self.batch_norm_1 = BatchNormalization()
		self.leaky_1 = LeakyReLU(alpha=0.2)

		# Block 2, Form: (16, 16, 64) -> (8, 8, 128)
		self.conv2d_2 = Conv2D(filters=128, kernel_size=5, strides=(2, 2), padding='same')
		self.batch_norm_2 = BatchNormalization()
		self.leaky_2 = LeakyReLU(alpha=0.2)

		# Block 3, Form: (8, 8, 128) - > (4, 4, 256)
		self.conv2d_3 = Conv2D(filters=256, kernel_size=5, strides=(2, 2), padding='same')
		self.batch_norm_3 = BatchNormalization()
		self.leaky_3 = LeakyReLU(alpha=0.2)

		# Ausgabeschicht, Form: (4, 4, 256) -> (1,)
		self.flatten = Flatten()
		self.output_layer = Dense(units=1, activation='linear')

	def call(self, input_tensor):
		x = self.input_layer(input_tensor)
		x = self.leaky_0(x)
		# (32, 32, 32)
		x = self.conv2d_1(x)
		x = self.batch_norm_1(x)
		x = self.leaky_1(x)
		# (16, 16, 64)
		x = self.conv2d_2(x)
		x = self.batch_norm_2(x)
		x = self.leaky_2(x)
		# (8, 8, 128)
		x = self.conv2d_3(x)
		x = self.batch_norm_3(x)
		x = self.leaky_3(x)      
		# (4, 4, 256)
		x = self.flatten(x)
		x = self.output_layer(x)
		# (1, )
		return x


# -------------------------------------------------  Fehlerfunktionen  -------------------------------------------------

# Die Fehlerfunktionen Kreuzentropie
kreuz_entropie = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_fehler_funktion(vorhersage_fake_bilder):
	# Entropie zwischen der Vorhersage des Diskriminators bei generierten Bildern (fake) und dem Ziel-Wert eins.
	return kreuz_entropie(tf.ones_like(vorhersage_fake_bilder), vorhersage_fake_bilder)                


def diskriminator_fehler_funktion(vorhersage_echte_bilder, vorhersage_fake_bilder):

	# Entropie zwischen der Vorhersage des Diskriminators bei Bildern des Datensatzes (echt) und dem Ziel-Wert eins.
	fehler_vorhersage_echte_bilder = kreuz_entropie(tf.ones_like(vorhersage_echte_bilder), vorhersage_echte_bilder)

	# Entropie zwischen der Vorhersage des Diskriminators bei generierten Bildern (fake) und dem Ziel-Wert null.
	fehler_vorhersage_fake_bilder = kreuz_entropie(tf.zeros_like(vorhersage_fake_bilder), vorhersage_fake_bilder)

	# Summe bilden, um die gesamte Entropie 
	gesamter_fehler = fehler_vorhersage_echte_bilder + fehler_vorhersage_fake_bilder
	
	return gesamter_fehler



# -------------------------------------------------  Hauptprogramm  --------------------------------------------------

# Datensatz laden
datensatz = datsensatz_laden_und_vorverarbeiten(
	name=DATENSATZ_NAME,
	batch_grösse=BATCH_GRÖSSE//2,
	bild_grösse=BILD_GRÖSSE)

# Netzwerke definieren
generator = Generator()
diskriminator = Diskriminator()

# Optimierungsverfahren definieren
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=GENERATOR_LERNRATE, beta_1=0.5)
diskriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=DISKRIMINATOR_LERNRATE, beta_1=0.5)

# Summary Writer
log_dir = 'logs/' + "Generative Adversarial Network"
file_writer = tf.summary.create_file_writer(log_dir)

# Mit diesen Funktionen wird der Durchschnitt des Fehlers während einer Epoche berechnet.
gen_fehler_durchschnitt = tf.keras.metrics.Mean('generator_fehler', dtype=tf.float32)
disk_fehler_durchschnitt = tf.keras.metrics.Mean('diskriminator_fehler', dtype=tf.float32)


# Trainingsschritt
def trainings_schritt(generator, diskriminator, datensatz_bilder, batch_grösse):
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disk_tape:

		# Generator generiert 32 Bilder
		eingabevektor = generator.generiere_eingabevektor(
			batch_grösse=batch_grösse, # 32 von 64 Bilder werden von Generator generiert.
			länge=100)
		generierte_bilder = generator(eingabevektor)


		# Diskriminator macht Vorhersage über die Wahrscheinlichkeit, dass die Bilder vom Datensatz stammen. 
		vorhersage_echte_bilder = diskriminator(datensatz_bilder) # 32 Bilder aus dem Datensatz 
		vorhersage_fake_bilder = diskriminator(generierte_bilder) # 32 Bilder von Generator

		# Diskriminator Optimierung
		diskriminator_fehler = diskriminator_fehler_funktion(vorhersage_echte_bilder, vorhersage_fake_bilder)
		diskriminator_gradienten = disk_tape.gradient(diskriminator_fehler, diskriminator.trainable_variables)
		diskriminator_optimizer.apply_gradients(zip(diskriminator_gradienten, diskriminator.trainable_variables))

		# Generator Optimierung
		generator_fehler = generator_fehler_funktion(vorhersage_fake_bilder)
		generator_gradienten = gen_tape.gradient(generator_fehler, generator.trainable_variables)
		generator_optimizer.apply_gradients(zip(generator_gradienten, generator.trainable_variables))

		return diskriminator_fehler, generator_fehler



def training(datensatz, epochen, batch_grösse, trainings_schritte_pro_epoche):

	template = "Epoche {}/{}, Generator Fehler: {:5.3f}, Diskriminator Fehler: {:5.3f}"
	
	
	for epoche in range(epochen):
		epoche += 1
		print(f"Epoche {epoche}/{epochen} ...", end='\r')

		for batch in datensatz.take(trainings_schritte_pro_epoche):
	
			disk_fehler, gen_fehler = trainings_schritt(generator, diskriminator, batch, batch_grösse=batch_grösse)
			disk_fehler_durchschnitt(disk_fehler)   # Speichert die Fehler des Diskriminators, um sie auszugeben.
			gen_fehler_durchschnitt(gen_fehler)     # Speichert die Fehler des Generators,um sie auszugeben.

		# Ausgeben der momentanen Epoche und der Fehler nach jeder Epoche
		print(template.format(
			epoche, epochen,
			gen_fehler_durchschnitt.result(),
			disk_fehler_durchschnitt.result()
		))
		
		with file_writer.as_default():
			tf.summary.scalar('generator_fehler', gen_fehler_durchschnitt.result(), step=epoche)
			tf.summary.scalar('diskriminator_fehler', disk_fehler_durchschnitt.result(), step=epoche)

			fake_bilder = generator(SEED)
			fake_bilder = (fake_bilder + 1) / 2
			fake_bilder = np.clip(fake_bilder, 0, 1)
			tf.summary.image("Generierte Bilder", data=fake_bilder, step=epoche)
		
		gen_fehler_durchschnitt.reset_states()
		disk_fehler_durchschnitt.reset_states()
	
	return generator, diskriminator



# Ausführen der Training-Funktion des GAN
generator, discriminator = training(datensatz, EPOCHEN, BATCH_GRÖSSE//2, TRAININGS_SCHRITTE_PRO_EPOCHE)

# Speichern der Gewichte
generator.save_weights('generator.h5')
diskriminator.save_weights('discriminator.h5')
