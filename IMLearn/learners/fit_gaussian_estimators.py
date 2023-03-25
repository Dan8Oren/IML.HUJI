import numpy as np
import plotly.graph_objects as go

from gaussian_estimators import UnivariateGaussian, MultivariateGaussian

#   ######//  Question 1  //######
ug = UnivariateGaussian()
mu, sigma = 10, 1
samples_amount = 1000
# Generate 1000 samples from the normal distribution with mean mu and
# standard deviation sigma
samples = np.random.normal(mu, sigma, samples_amount)

ug.fit(samples)
print((ug.mu_, ug.var_))

#   ######//  Question 2  //######
all_mus = []
for i in range(10, 1000, 10):
    all_mus.append(np.abs(UnivariateGaussian().fit(samples[:i]).mu_ - mu))
fig2 = go.Figure(go.Scatter(x=list(range(len(all_mus))), y=all_mus),
                 dict(
                     title="Difference of estimated & true expectation as a "
                           "function of sample size",
                     xaxis_title="Sample size",
                     yaxis_title="Difference of the expectation"))
fig2.update_traces(mode='markers', marker=dict(line_width=1, symbol='circle',
                                               size=6))
# fig2.write_image("Question2.png")

#   ######//  Question 3  //######
pdfs = ug.pdf(samples)
sample_pdf_tuples = []
for i in range(samples_amount):
    sample_pdf_tuples.append((samples[i], pdfs[i]))

sample_pdf_tuples.sort(key=lambda t: t[1])  # sort by pdf value
fig3 = go.Figure(go.Scatter(x=[x[0] for x in sample_pdf_tuples],
                            y=[x[1] for x in sample_pdf_tuples]),
                 dict(title="Empirical PDF of the fitted model",
                      xaxis_title="Sample value",
                      yaxis_title="PDF value"))
fig3.update_traces(mode='markers', marker=dict(line_width=1, symbol='circle',
                                               size=6))
# fig3.write_image("Question3.png")

#   ######//  Question 4  //######

multivariate_mu = [0, 0, 4, 0]
multivariate_cov = np.array([
    [1, 0.2, 0, 0.5],
    [0.2, 2, 0, 0],
    [0, 0, 1, 0],
    [0.5, 0, 0, 1]
])
multivariate_samples = np.random.multivariate_normal(mean=multivariate_mu,
                                                     cov=multivariate_cov,
                                                     size=samples_amount)
mg = MultivariateGaussian()
mg.fit(multivariate_samples)
print(mg.mu_)
print(mg.cov_)

#   ######//  Question 5  //######

result = np.zeros((200, 200))
row = 0
f1_values = np.linspace(-10, 10, 200)
f3_values = np.linspace(-10, 10, 200)
for f1 in f1_values:
    col = 0
    for f3 in f3_values:
        temp_mu = [f1, 0, f3, 0]
        result[row, col] = MultivariateGaussian.log_likelihood(
            np.array(temp_mu), multivariate_cov,
            multivariate_samples)
        col += 1
    row += 1
heatmap = go.Heatmap(x=f3_values, y=f1_values, z=result)
fig5 = go.Figure(data=heatmap, layout=dict(
    title=r"$\text{Multivariate gaussian log-likelihood as function of "
          r"expectation }(\mu)$",
    xaxis_title=r"$\text{f3 values }(\mu_3)$",
    yaxis_title=r"$\text{f1 values }(\mu_1)$"))

# fig5.write_image("Question5.png")

#   ######//  Question 6  //######
min_indices = np.unravel_index(np.argmax(result, axis=None), result.shape)
print(
    [round(f3_values[min_indices[1]], 3), round(f1_values[min_indices[0]], 3)])
