# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
try:
    import tensorflow as tf
    import tensorboard.plugins.mesh.summary as meshsummary
except ImportError:
    print('tensorflow is not installed.')
import numpy as np
import scipy.misc


try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class TfLogger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.compat.v1.summary.FileWriter(log_dir)

        # Camera and scene configuration.
        self.config_dict = {
            'camera': {'cls': 'PerspectiveCamera', 'fov': 75},
            'lights': [
                {
                    'cls': 'AmbientLight',
                    'color': '#ffffff',
                    'intensity': 0.75,
                }, {
                    'cls': 'DirectionalLight',
                    'color': '#ffffff',
                    'intensity': 0.75,
                    'position': [0, -1, 2],
                }],
            'material': {
                'cls': 'MeshStandardMaterial',
                'metalness': 0
            }
        }

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.compat.v1.Summary.Image(encoded_image_string=s.getvalue(),
                                                 height=img.shape[0], width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.compat.v1.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def mesh_summary(self, tag, vertices, faces=None, colors=None, step=0):

        """Log a list of mesh images."""
        if colors is None:
            colors = tf.constant(np.zeros_like(vertices))
        vertices = tf.constant(vertices)
        if faces is not None:
            faces = tf.constant(faces)
        meshes_summares=[]
        for i in range(vertices.shape[0]):
            meshes_summares.append(meshsummary.op(
                tag, vertices=vertices, faces=faces, colors=colors, config_dict=self.config_dict))

        sess = tf.Session()
        summaries = sess.run(meshes_summares)
        for summary in summaries:
            self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

