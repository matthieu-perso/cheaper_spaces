### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ bc1220d8-3e5f-11ee-1966-0b1b5ba18a16
begin
	using CSV
	using DataFrames
	using Distances
	using Gadfly
	using Statistics
	using StatsBase
	using GLM
	using MultivariateStats
	using RCall
	using Images
	using Embeddings
end

# ╔═╡ 85c8b5a1-bdca-4d55-beba-537add4ed507
wd = "/Users/matthieu/Downloads/reembeddings/";

# ╔═╡ aa460f57-0bc6-444a-9118-66842c9f66bf
begin
	labels = CSV.read(wd * "labels.csv", DataFrame, header=false)
	pairs = CSV.read(wd * "henley.csv", DataFrame, header=false)
	coords = CSV.read(wd * "coords.csv", DataFrame)
	inference = CSV.read(wd * "inference.csv", DataFrame)
end;

# ╔═╡ cbe83826-1f31-41ee-a15c-2a725b7760a0
md"""
In Douven, Verheyen, Elqayam, Gärdenfors, and Osta-Vélez, "Similarity-based reasoning in conceptual spaces" (_Frontiers in Psychology_, forthcoming), we let participants arrange twenty items (mammals) in a two-dimensional space. The results were interpreted as yielding the participants' personal mammal spaces. By averaging over the distances among the items and then applying multidimensional scaling to the averages, we obtained a general mammal space. The construction is repeated below.
"""

# ╔═╡ 8aa433bb-3775-4bc3-8ced-f8d3a78e37b4
crds = Array{Float64,3}(undef, size(inference, 1), 2, 20); # number of Ps, x and y coords, 20 items

# ╔═╡ aac20471-7c34-40c2-8253-27cae3d9cfac
for i in 1:20
    crds[:, 1, i] = coords[:, (3:2:41)[i]] ./ coords[:, 1]
    crds[:, 2, i] = coords[:, (4:2:42)[i]] ./ coords[:, 2]
end

# ╔═╡ 703c9440-6d2b-411a-9e82-4c42cb8b6c43
crds_per_prt = Array{Float64,3}(undef, 20, 2, size(inference, 1));

# ╔═╡ 43a0c9a6-c898-40dc-9595-262fce56a8c4
for i in 1:size(inference, 1), j in 1:20
    crds_per_prt[j, :, i] = crds[i, :, j]
end

# ╔═╡ 0add1b77-e3b9-405b-880c-c01e5eacc70a
dists = [ pairwise(Euclidean(), crds_per_prt[:, :, i], dims=1) for i in 1:size(inference, 1) ];

# ╔═╡ 954a9af2-f800-450b-8d50-55ddea080d2d
dst_mtx = mean(dists, dims=1)[1];

# ╔═╡ aa1a9fb4-c62f-4c04-8d50-90ee66636983
# ╠═╡ show_logs = false
R"library(vegan)";

# ╔═╡ 25e59a93-d5cc-42cd-80c3-735f05b353f8
@rput dst_mtx;

# ╔═╡ f8b89ca5-2efe-4504-9185-698814f65354
mds1 = fit(MDS, dst_mtx; maxoutdim=2, distances=false)

# ╔═╡ 4d66c955-be09-4d42-8c74-7dbc82c17303
preds1 = predict(mds1);

# ╔═╡ dc5b7e52-26be-45f3-8f84-ea744e1d0cad
md"Aggregate mammal space:"

# ╔═╡ b1177c92-dded-4d2d-9fb3-b5f799d4333f
plot(x=preds1[1, :], y=preds1[2, :], label=labels.Column1, Geom.point, Geom.label(hide_overlaps=false), Guide.xlabel(nothing), Guide.ylabel(nothing), Coord.cartesian(aspect_ratio=1))

# ╔═╡ fa5477e8-80a8-4253-a167-e216d12d200c
md"""
Problem: In the _Frontiers_ paper, a Visual Arrangement Task was used to obtain the participants' mammal spaces. That is only possible if one is reasonably certain that a two-dimensional space will be adequate given the materials (it can be adequate even if one suspects that a three-dimensional space would be better). The more common and more general procedure is to record participants' pairwise similarity judgments for the materials and then use multi-dimensional scaling or a similar technique to arrive at a similarity space. But this procedure can be prohibitively expensive. More importantly, the common procedure as well as the VAT share the problem that they normally do not permit generalizations to new materials.

To illustrate this problem, note that in the _Frontiers_ paper only a subset of Henley's set of mammals was used, to reduce the complexity of the task for the participants. Left out were all of the following:
"""

# ╔═╡ 1ee0c8e1-f930-4350-83d9-39e6d304b7b1
left_out = ["monkey", "leopard", "bear", "antelope", "raccoon", "deer", "elephant", "squirrel", "chipmunk"];

# ╔═╡ 0baf238b-bbc0-4e5a-9089-ebedc05149b9
md"""
Where, in the above space, should we place these? For some we can guess where they would end up _approximately_ (e.g., the elephant probably not too far from the gorilla and the tiger) and the squirrel probably not too far from the mouse, rat, and rabbit. But the raccoon? In any event, better than approximating the locations is not possible, given that the dimensions, though not entirely uninterpretable (the _x_-axis seeming somewhat correlated with size, for instance), certainly do not allow for a metric interpretation, which would otherwise allow to measure the relevant features of the mammals left out and place them into the space accordingly. 
"""

# ╔═╡ 90ce10d6-1715-49f7-8595-cfd54250868c


# ╔═╡ d92a132f-c765-4885-8a1f-181e03ca5834
md"### GPT spaces"

# ╔═╡ faf447b5-75d3-418b-98da-3f4aa23fa97a
md"""
We re-used the materials from "Similarity-based reasoning in conceptual spaces" and entered these into ChatGPT (GPT-4), asking it to generate a similarity matrix for the set. We explicitly asked it to use a scale from 0 to 10, with 0 indicating maximum dissimilarity and 10 indicating maximum similarity, and also to assign 10 only in the case of identity (without this request, it on one occasion assigned 10 to the `goat-sheep` pair as well as to `rat-mouse` pair; _with_ the request, these are rated 9). In fact, we repeated this procedure at different points in time, finding slight differences between the responses. 
"""

# ╔═╡ 0fc7b8ef-af3e-453e-9415-d4c755a9b45d
begin
	sims1 = CSV.read(wd * "gpt1.csv", DataFrame; header=true)
	sims2 = CSV.read(wd * "gpt2.csv", DataFrame; header=true)
	sims3 = CSV.read(wd * "gpt3.csv", DataFrame; header=true)
	sims4 = CSV.read(wd * "gpt4.csv", DataFrame; header=true)
	sims5 = CSV.read(wd * "gpt5.csv", DataFrame; header=true)
end;

# ╔═╡ bcb6af3b-d094-4066-9c4f-1f5d022d404a
labs = names(sims1);

# ╔═╡ 8017a28f-b0b7-4d77-b8c4-1a61c35a95ee
begin
	dst1 = pairwise(Euclidean(), Matrix(sims1))
	dst2 = pairwise(Euclidean(), Matrix(sims2))
	dst3 = pairwise(Euclidean(), Matrix(sims3))
	dst4 = pairwise(Euclidean(), Matrix(sims4))
	dst5 = pairwise(Euclidean(), Matrix(sims5))
end;

# ╔═╡ aee253e3-7e48-41bd-a98d-aeb055ce94f6
mds2 = fit(MDS, Float64.(Matrix(sims1)); maxoutdim=2, distances=false)

# ╔═╡ 2dcf7754-8f58-4db5-a6cf-140ce2d5fe8d
preds2 = predict(mds2);

# ╔═╡ 4ea7cb60-d418-4602-87ef-c007bf2469f5
md"This space is obtained by again applying multidimensional scaling, now to one of the distances matrices we were given by ChatGPT:"

# ╔═╡ 722142df-a242-483f-b8c2-b217b2a7895a
plot(x=preds2[1, :], y=preds2[2, :], label=labs, Geom.point, Geom.label(hide_overlaps=false), Guide.xlabel(nothing), Guide.ylabel(nothing), Coord.cartesian(aspect_ratio=1))

# ╔═╡ 5f8ef85b-4c6e-4997-9af8-8b72aa0fb79f
@rput dst1;

# ╔═╡ a7a64ed0-8b66-4036-9bdd-2a99cd3802c0
@rput dst2;

# ╔═╡ e7255ef1-bad0-4655-bcdc-3cc62329c574
@rput dst3;

# ╔═╡ 8ef0abcf-622d-472f-8665-ba6bd382f423
@rput dst4;

# ╔═╡ cd562d4e-d23e-4721-860d-d3d5c761abb0
@rput dst5;

# ╔═╡ 04cdab51-28e7-4868-be7e-acefb3b300c2
md"Now we are going to see how strongly (if at all) the distance matrices from ChatGPT correlate with the one from the participants. We use the Procustes test for this, which gives us the highest correlation that can be obtained by scaling and rotating one of the spaces relative to the other."

# ╔═╡ 2f2e73ca-af62-4ac9-9a87-edd4f8dca689
md"Note that the items do not occur in the same order in the dataset from Douven et al. and in the matrices from ChatGPT, so we first reorder the latter."

# ╔═╡ 9dd11ecc-ea4e-45a9-91e5-e3e741e345b0
perm = [ findfirst(==(labels.Column1[i]), labs) for i in 1:20 ];

# ╔═╡ 34b2a1da-b788-4c05-8204-3358ead5814e
@rput perm;

# ╔═╡ a80ea0e9-d7a3-4ce8-b495-535dd03f4346
R"protest(dst_mtx, dst1[perm, perm])"

# ╔═╡ 4a2e3549-eac4-4137-8dab-dd947390e047
R"protest(dst_mtx, dst2[perm, perm])"

# ╔═╡ f695a9a5-646c-4f17-868a-6f145415b831
R"protest(dst_mtx, dst3[perm, perm])"

# ╔═╡ 182732a1-f60a-47c5-a65f-3e55aa41c4c1
R"protest(dst_mtx, dst4[perm, perm])"

# ╔═╡ 336550e8-c077-4c74-b90a-d6d0da8a2ab6
R"protest(dst_mtx, dst5[perm, perm])"

# ╔═╡ 5d4ac17c-2d35-4ab7-b6ec-7f6aa36c1ec5
md"Thus, we consistently find a very high correlation, in two cases even a correlation of .92. The correlations are also highly significant in all cases."

# ╔═╡ 3652a28d-fa65-47e1-82b2-84024477ecee
md"It is also worth checking the consistency of ChatGPT by looking at the correlation between pairs of matrices we obtained from it. (Note that here it is not necessary to reorder the matrices.)"

# ╔═╡ 74233fe9-8b20-4f79-9ec8-f49d9c8218e6
R"protest(dst1, dst4)"

# ╔═╡ c19c4718-a280-4fbe-add8-81ee41ee234b
md"Here, too, we find highly significant and consistently high correlations."

# ╔═╡ ce4eb397-0ea0-44c0-9447-f86837e7ee4c
md"### BARD spaces"

# ╔═╡ 353c8c7d-7c95-4f3e-919d-9f31c0b39ab2
md"We repeated the same procedure for BARD. However, an immediate problem was that BARD typically did _not_ return a symmetric similarity matrix. We tried twelve times and only got a symmetric matrix on two of those trials. We report those."

# ╔═╡ 9319a596-519f-4f3c-ac4e-f77335e4c708
begin
	bsims1 = CSV.read(wd * "bard1.csv", DataFrame; header=true)
	bsims2 = CSV.read(wd * "bard2.csv", DataFrame; header=true)
end;

# ╔═╡ c11d45ee-469e-4343-8579-3ddfd6f022ac
begin
	bdst1 = pairwise(Euclidean(), Matrix(bsims1))
	bdst2 = pairwise(Euclidean(), Matrix(bsims2))
end;

# ╔═╡ 604a02fb-4a05-4d69-b250-0e9bfcd98a89
bmds2 = fit(MDS, Float64.(Matrix(bsims1)); maxoutdim=2, distances=false)

# ╔═╡ 133fde06-00d7-41f9-b4fc-7502dd8f22dd
bpreds2 = predict(bmds2);

# ╔═╡ 90e67763-9aa5-4f90-a5f0-38513b310cc6
plot(x=bpreds2[1, :], y=bpreds2[2, :], label=labs, Geom.point, Geom.label(hide_overlaps=false), Guide.xlabel(nothing), Guide.ylabel(nothing), Coord.cartesian(aspect_ratio=1))

# ╔═╡ 57db18a6-d35e-45ac-bc8b-b77699c474ce
@rput bdst1;

# ╔═╡ 783bf0f0-0fd1-408b-8e8c-6ea7490bda16
@rput bdst2;

# ╔═╡ 967985f9-6685-4b65-8d2c-521acc93cbed
R"protest(dst_mtx, bdst1[perm, perm])"

# ╔═╡ af7afb1c-51e8-41e6-8a6a-1d4da586864d
R"protest(dst_mtx, bdst2[perm, perm])"

# ╔═╡ 14cf911b-6346-45da-9509-2f0de0e5b6ac
md"We see that for BARD the results are much worse."

# ╔═╡ a1b42e8f-331b-44a4-94c2-ea801a3209a0
R"protest(bdst1, bdst2)"

# ╔═╡ 310c5e3e-f7b2-425f-886c-bb184dc159df
md"Interesting to see, also, that the two valid result we got do not even correlate so highly with each other."

# ╔═╡ 0b9e4ba3-eb25-43de-b3e1-8f99f47a56a2
md"### Embeddings"

# ╔═╡ baada914-8cd6-4499-9bbe-53ff3f813ffc
md"The following is an attempt to get the similarities, and subsequently the mammal space, in a still cheaper way, namely, by extracting the vectors corresponding to the various mammal predicates from pre-trained transformer models. More specifically, we will want to extract the spaces from so-called word embeddings, which represent words as points in a high-dimensional space, with words being more similar located closer to each other. Word embeddings are usually trained via unsupervised learning, though exactly how differs for the different models. Note that, precisely because the words are represented by high-dimensional vectors, we will have to apply a dimension reduction technique to them if we want to obtain a similarity space that we can visualize and that we can build a conceptual space upon."

# ╔═╡ 835e1e12-2184-435e-a490-230d077efd16
md"##### ELMo"

# ╔═╡ fab962cd-45cc-46cf-bd6b-a36a91813273
md"The best transformer models are currently closed source. But various computer languages do give access to a number of powerful models. We used _Mathematica_ to access the ELMo (Embeddings from Language Models) model, via the function `NetModel[\"ELMo Contextual Word Representations Trained on 1B Word Benchmark\"]`, which has the following architecture:"

# ╔═╡ 6f188c70-f27c-4189-bb58-20d253e08c57
load(wd * "model.png")

# ╔═╡ e244fa4a-6f34-4e13-99c5-978942de18e2
md"Once that model is loaded, we obtain 1054-dimensional vectors representing, for instance, 'cat' via this function: `model[TextElement[{\"cat\"}]][[\"Embedding\", 1]]`. This was done for all twenty mammals from the materials of the _Frontiers_ study. Finally, we used MDS to reduce the dimensionality of these data to just two, which gave the following solution:"

# ╔═╡ d817ae43-fdfd-40c5-85e8-900d300c889e
math_mds = CSV.read(wd * "mathematica_mds.csv", DataFrame; header=false);

# ╔═╡ d8527fe6-3ea6-4f51-95d5-f663cf5aafc9
plot(x=math_mds[:, 1], y=math_mds[:, 2], label=labs, Geom.point, Geom.label(hide_overlaps=false), Guide.xlabel(nothing), Guide.ylabel(nothing), Coord.cartesian(aspect_ratio=1))

# ╔═╡ 3b8f0266-4900-4b68-b316-9cbf303cb5ae
math_dst = pairwise(Euclidean(), Matrix(math_mds)');

# ╔═╡ d8eba36c-263a-47f4-97de-083900ac45e6
@rput math_dst;

# ╔═╡ b375285d-44ff-409a-99cb-79214770a70c
R"protest(dst_mtx, math_dst[perm, perm])"

# ╔═╡ ea0d2ff6-576f-4eae-aec1-39d07dc9a97a
md"The correlation is significant and at least moderate. This result is still somewhat encouraging, given that this is by no means the best transformer model for this purpose."

# ╔═╡ 4a6db4ad-9dc4-4b75-a700-f6855da6f352
md"#### Going 3D?"

# ╔═╡ fe30cb66-2c23-4c1d-bdf4-62945b1bc4e8
md"For the _Frontiers_ paper, a Visual Arrangement Task was used, thereby necessarily limiting spaces to two dimensions. For the embeddings, we might as well reduce the 1054-dimensional vectors to three-dimensional data, then measure distances in the resulting space, and finally compare those distances to the distances we got from the aggregate space reported in the _Frontiers_ paper. The reduction to three dimensions was done again in _Mathematica_."

# ╔═╡ 4d71aac6-2c62-439b-abe1-943b81d3d78a
math_mds_3d = CSV.read(wd * "mathematica_mds_3d.csv", DataFrame; header=false);

# ╔═╡ 4c90e1ac-9c11-4278-b800-2f29b85841a2
math_dst_3d

# ╔═╡ 79e14648-19a0-43a6-beba-a5e1de818cce
@rput math_dst_3d;

# ╔═╡ ace7343d-842a-4f59-a5bd-33cd62079b9c
R"protest(dst_mtx, math_dst_3d[perm, perm])"

# ╔═╡ f8f299c4-39d1-489a-800a-d67529753395
md"This actually does improve the correlation by quite a bit."

# ╔═╡ 80b3b3d3-02f4-4d79-aaea-fcef3c1ad8d9
md"#### More dimensions still"

# ╔═╡ 45472a6a-c7d7-4a2e-87f0-4659e85ecd0c
md"What if we reduce the 1054-dimensional vectors to a five-dimensional space?"

# ╔═╡ 1baf1101-aaaf-453c-aa1f-1c3cb932e1a5
math_mds_5d = CSV.read(wd * "mathematica_mds5d.csv", DataFrame; header=false);

# ╔═╡ fdea0e39-2314-4b80-998d-47a756ecac86
math_dst_5d = pairwise(Euclidean(), Matrix(math_mds_5d)');

# ╔═╡ 0e2f3868-c1f7-4a4d-b16a-c794fffc4060
@rput math_dst_5d;

# ╔═╡ 59babe1a-a922-4ddb-858e-ad25153997b0
R"protest(dst_mtx, math_dst_5d[perm, perm])"

# ╔═╡ d88d90f6-f054-4ba7-8a27-6d2a012fcabb
md"Only slightly better. How about we measure distances in the original 1054-dimensional vector space?"

# ╔═╡ 32b991cc-042d-4e69-8884-1157a0b64d03
vecs = CSV.read(wd * "vecs1054d.csv", DataFrame; header=false);

# ╔═╡ 6a690490-d1a8-4795-89c7-01759b9ff3f3
vecs_dst = pairwise(Euclidean(), Matrix(vecs)');

# ╔═╡ ffc0df4d-2ba3-4071-9bc0-6a698f400413
@rput vecs_dst;

# ╔═╡ 8fb3a9fa-e6f0-44e1-8191-579fa1b2999d
R"protest(dst_mtx, vecs_dst[perm, perm])"

# ╔═╡ 26d79d37-f6bc-454e-af54-039f40f23e25
md"Again slightly better."

# ╔═╡ f17c0070-5413-4e18-b22b-fd8187cda2a4
md"According to the literature, it is not wrong to measure distances in the embedding (a high-dimensional vector space) using a Euclidean metric, but most researchers recommend using the cosine distance. We used this as well as some other metrics (e.g., the Manhattan metric) and using the cosine distance yielded a considerable improvement."

# ╔═╡ f319b01c-3c4a-4414-8724-c2dc60819a41
vecs_dst_cd = pairwise(CosineDist(), Matrix(vecs)');

# ╔═╡ 86cb9d38-6498-45db-9e44-11f5872f7b7e
@rput vecs_dst_cd;

# ╔═╡ 5ffe2933-554a-43f9-9c28-fe7e1aa2bb78
R"protest(dst_mtx, vecs_dst_cd[perm, perm])"

# ╔═╡ 4a127d95-a24a-4f2e-b362-d01342140558
md"This is starting to look really good."

# ╔═╡ a4fbf469-bb7a-41d4-92ff-9d406ab206b8
md"##### FastText"

# ╔═╡ 2b2e24b9-d583-42e9-9165-ec5bf891fe40
md"We access the `FastText` model via the `Embeddings.jl` package. (This package is supposed to give access also to the `Word2Vec` model, but this is currently not working.)"

# ╔═╡ b0547f88-b11a-4775-96c7-00903ac3dbed
begin
	const embtable = load_embeddings(FastText_Text)

	const get_word_index = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))

	function get_embedding(word)
    	ind = get_word_index[word]
    	emb = embtable.embeddings[:,ind]
    	return emb
	end
end

# ╔═╡ 4513c56e-c184-4bf1-bdbc-74ada2e89900
ft_embs = get_embedding.(labels.Column1);

# ╔═╡ 255ff2b1-8cda-48d2-b2d6-a29009131f27
md"Here, too, we use both the Euclidean metric and the cosine distance."

# ╔═╡ 485ebe6b-023c-423a-8be7-8fe6c964a416
ft_dst = pairwise(Euclidean(), ft_embs);

# ╔═╡ 0efb986a-3701-48d1-9823-a1522a2c66c0
@rput ft_dst;

# ╔═╡ 6ef3626b-37e9-4602-9cb2-a23067f8a73f
R"protest(dst_mtx, ft_dst)"

# ╔═╡ 975da245-9ce2-4397-85e5-80b1d4a4c1ce
md"This is an excellent result: a highly significant correlation of .84!"

# ╔═╡ 0264d362-b1bc-46ae-92fd-914359d27e3d
ft_dst_cd = pairwise(CosineDist(), ft_embs);

# ╔═╡ c408f9cc-a21f-40a0-94e9-23f7c9dc7673
@rput ft_dst_cd;

# ╔═╡ abfd806b-bf99-4cd3-a33c-fc6690265c70
R"protest(dst_mtx, ft_dst_cd)"

# ╔═╡ 67265e5a-42da-479c-ab05-3148f3423243
md"This is a highly significant correlation of .89!"

# ╔═╡ 07b77a36-ddb9-47cb-998a-f3e6e2c2a954
md"As a kind of sanity check, let us see what a two-dimensional space based on these distances looks like."

# ╔═╡ 336ec162-a6ad-49b7-8b97-1be5487a3b13
mds_ft = fit(MDS, ft_dst_cd; maxoutdim=2, distances=true)

# ╔═╡ d6e7ee63-1be4-482f-b5c0-9c558f37bf22
prd_ft = predict(mds_ft);

# ╔═╡ d90ab635-3c95-44ab-ac4c-902a1585bad6
plot(x=prd_ft[1, :], y=prd_ft[2, :], label=labels.Column1, Geom.point, Geom.label(hide_overlaps=false), Guide.xlabel(nothing), Guide.ylabel(nothing), Coord.cartesian(aspect_ratio=1))

# ╔═╡ 4ba3eb73-5c64-458d-9b7d-8d1ee8c844bb
md"This makes sense!"

# ╔═╡ 47b34b3d-7d46-482b-9b0c-07a09f07be5c
md"""
Remember that in the _Frontier_ paper mammals were left out from Henley's set. We said at the beginning of this document that we could not reliably put the left out mammals into the space on the basis of their features. But the excellent result we just got suggests that we can use the `FastText` model for this. Just repeat the above procedure but now for the entire set. 
"""

# ╔═╡ 426ed389-7df8-4b29-b8e8-b418b2815cc4
full_embs = get_embedding.(vcat(labels.Column1, left_out));

# ╔═╡ 0dea6480-5116-4458-bafd-b4ea1afa6214
full_dst_cd = pairwise(CosineDist(), full_embs);

# ╔═╡ b0a2e043-7b40-4175-8f28-0a08928edb02
mds_full = fit(MDS, full_dst_cd; maxoutdim=2, distances=true)

# ╔═╡ 3b33cf52-37ae-4cbc-83d3-ecc09f9f9e2b
prd_full = predict(mds_full);

# ╔═╡ 399b874d-bee4-40af-a66e-8f26de2b859b
plot(x=prd_full[1, :], y=prd_full[2, :], label=vcat(labels.Column1, left_out), Geom.point, Geom.label(hide_overlaps=false), Guide.xlabel(nothing), Guide.ylabel(nothing), Coord.cartesian(aspect_ratio=1))

# ╔═╡ 79668bfc-e5d5-4bda-b54d-c0b34749ca6a
md"Again, this makes a _lot_ of sense!"

# ╔═╡ 5224ef88-c0c6-4db3-86b9-90a610a7aa79
md"##### GloVe"

# ╔═╡ dd8aecbd-c0d2-4ef7-aadf-09e32663303d
const glove = load_embeddings(GloVe{:en}, 3, max_vocab_size=50000);

# ╔═╡ 5a5eb6a5-374d-47d8-88f3-a895404faeb7
const get_word_ind_glove = Dict(word=>ii for (ii,word) in enumerate(glove.vocab));

# ╔═╡ 56090994-128a-4476-b506-7920a938180a
function get_embedding_glove(word)
   	ind = get_word_ind_glove[word]
   	emb = glove.embeddings[:,ind]
    return emb
end

# ╔═╡ eedd241b-9ad9-495f-b75d-74ed9c05528f
gl_embs = get_embedding_glove.(labels.Column1);

# ╔═╡ 6b695d46-3eb5-4961-8181-af517341db5e
md"We only report the result for the cosine distance, which was again considerably better than the results obtained using other metrics."

# ╔═╡ 185a8efd-075f-4991-b3e8-b2c25cb6e830
gl_dst = pairwise(CosineDist(), gl_embs);

# ╔═╡ 519669dc-994c-4779-985c-f4d8e67aeb09
@rput gl_dst;

# ╔═╡ 29cbb72a-abf6-43f2-a4d8-0a242c460058
R"protest(dst_mtx, gl_dst)"

# ╔═╡ 67a448ba-c544-4b66-9614-9fe0ef0eda80
md"Not as good as the `FastText` model, but still quite good."

# ╔═╡ d1747ce8-4865-414a-917c-68b57120fcce


# ╔═╡ 3648bbd7-6b08-435e-9cac-693338d8486e
md"#### Contrary to the models above, Bert uses the transformer architecture which has recently seen considerable sucess. 

Bert"

# ╔═╡ fab8dd11-bb25-4d3b-8d5b-d3ef78e2c4cc
bert_non_scaled = CSV.read(wd * "bert-base-uncased_distances.csv", DataFrame; header=true);


# ╔═╡ f2c46501-cfa2-4445-b15b-f72fdd419aa4
bert_cleaned = select!(bert_non_scaled, Not(1))  # Removes the first column


# ╔═╡ 53d51396-5b22-4d79-a37b-0b91021a1b6d
@rput bert_cleaned;

# ╔═╡ 6b989965-bd57-4d11-b26e-486d71966fc8
# ╠═╡ disabled = true
#=╠═╡
mat_base = Matrix{Float64}(bert_cleaned)

  ╠═╡ =#

# ╔═╡ 21e28b5a-09da-47db-95e2-3dfe5b28d281
R"protest(dst_mtx, bert_cleaned[perm, perm])"

# ╔═╡ 570b0213-0c7e-4f42-91c0-5ca00dfeb2fd
md"Using the Bert model gives us worse - but still significant results compared to the above static embeddings. This is probably due to the contextual nature of Bert embeddings  "

# ╔═╡ a85440cf-110b-4459-adcd-a45661aaa17a
md"Let us now see newer models - we will look at distilled model of much smaller size, as well as newer archtitectures.  "

# ╔═╡ 09521601-3be8-403c-b78c-e9697e30d775
distilbert = CSV.read(wd * "distilbert-base-uncased_distances.csv", DataFrame; header=true);


# ╔═╡ 8314fd26-c7e9-46ce-8d43-13ea6572487b
distilbert_cleaned = select!(distilbert, Not(1))  # Removes the first column


# ╔═╡ d558a989-b2b6-46a8-8d3d-c13dfa6f1cb1
@rput distilbert_cleaned;

# ╔═╡ 6096a53f-f5b4-42d7-9c22-9455f18f86f2
distilbert_base = Matrix{Float64}(distilbert_cleaned)


# ╔═╡ 515738e6-868c-43dd-a063-c0084ee7198f
R"protest(dst_mtx, bert_cleaned[perm, perm])"

# ╔═╡ e98cdc7f-8e81-4298-87e1-c7769b0c3878
md"Interestingly, a smaller base models gives us the same correlation, which mirrors the results obtained by the Distilbert authors. Even thought the model is much smaller it performs as well on downstream tasks."

# ╔═╡ 9b6b84e1-f9eb-49cf-8d7f-248af694226b
md"

#### RoBertA"

# ╔═╡ d9259412-4ccb-447f-a34b-06dc20659baf
roberta = CSV.read(wd * "roberta-base_distances.csv", DataFrame; header=true);


# ╔═╡ 013d1401-171a-4219-af04-c5e7ef002e5c
roberta_cleaned = select!(roberta, Not(1))  # Removes the first column


# ╔═╡ e2deebe4-fee7-49dc-96d2-558c24af5c40
@rput roberta_cleaned;

# ╔═╡ d0d411be-75f2-4126-8f64-2c93b43fe365
R"protest(dst_mtx, roberta_cleaned[perm, perm])"

# ╔═╡ 6526864b-fe7b-40cf-b81e-e156e09830f7
md"
The above were native transformer models and returned encouraging results, but considerably underneath the static embedding aboves. Perhaps native embedding models would work better. Let us start with the MPNet model from Sentence Transformer


MPNet is a variant of the Transformer architecture we've seen above that combines elements from both BERT and XLNet. Specifically, it integrates BERT's masked language modeling objective with XLNet's permutation-based training approach. This combination allows MPNet to learn flexible, context-dependent representations.

The loss function for MPNet, \( \mathcal{L}_{\text{MPNet}} \), can be seen as a hybrid that incorporates masked language modeling (like in BERT) and permuted sequences (like in XLNet). The mathematical expression for the loss function is given by:

\[
\mathcal{L}_{\text{MPNet}} = -\frac{1}{N} \sum_{i=1}^{N} \log p(x_i | \text{Permuted Context}_i; \theta)
\]

In this equation:

- \( \mathcal{L}_{\text{MPNet}} \) is the loss to be minimized.
- \( N \) is the total number of masked tokens.
- \( x_i \) is the \(i\)-th masked token.
- \( \text{Permuted Context}_i \) refers to a sequence of surrounding tokens that are permuted.
- \( \theta \) represents the model parameters.

The permutation in \( \text{Permuted Context}_i \) allows the model to predict each token based on a context that has been dynamically rearranged, making the model more flexible in learning contextual representations.

This added element could prove useful in conceptual space approximation. Let us experiment. 

#### Sentence transformers"

# ╔═╡ 7105694b-2121-445d-a4a8-79e74f01ee96
mpnet = CSV.read(wd * "all-mpnet-base-v2_distances.csv", DataFrame; header=true);

# ╔═╡ 67ec6b84-9f87-4ab0-8994-fa714e1a23e8
mpnet_cleaned = select!(mpnet, Not(1))  # Removes the first column


# ╔═╡ d1b0f02a-5924-4bac-8a84-29b167aed3e5
@rput mpnet_cleaned;

# ╔═╡ 80efe2f1-2df5-42c9-8be5-f9a373037bbd
R"protest(dst_mtx, mpnet_cleaned[perm, perm])"

# ╔═╡ e8d97c67-2cfe-4ee3-a65a-355deac7f46d
md"The results for the embedding models are much better than for the native transformer models. This can be in part be explained by the training approach. 

Given that sentence transformer is also a contextual embedding, this adds nuance to our previous analysis. 

#### Ada  002 Embeddings"

# ╔═╡ fa5a8d29-531b-40d0-8953-9acbd7774754
ada = CSV.read(wd * "text-embedding-ada-002_distances.csv", DataFrame; header=true);

# ╔═╡ 14b7ef6d-7a76-4e14-94e0-1a9b0ac9e813
ada_cleaned = select!(ada, Not(1))  # Removes the first column


# ╔═╡ 4d14f654-1d51-48e7-86a8-e046ee935daf
@rput ada_cleaned;

# ╔═╡ a25a9b13-a7c8-4f1e-a79b-c78e2b2e2e6e
R"protest(dst_mtx, ada_cleaned[perm, perm])"

# ╔═╡ 24276b2f-d710-4798-a16d-f1bbbd418c81
md"The results here are also quite encouraging and show similar results as the MPNET results above."

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
Embeddings = "c5bfea45-b7f1-5224-a596-15500f5db411"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
Gadfly = "c91e804a-d5a3-530f-b6f0-dfbca275c004"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
CSV = "~0.10.11"
DataFrames = "~1.6.1"
Distances = "~0.10.9"
Embeddings = "~0.4.1"
GLM = "~1.8.3"
Gadfly = "~1.4.0"
Images = "~0.26.0"
MultivariateStats = "~0.10.2"
RCall = "~0.13.16"
StatsBase = "~0.33.21"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "24fbc45d01f0cb55799bd679ba26849f21a71902"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f83ec24f76d4c8f525099b2ac475fc098138ec31"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.4.11"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e5f08b5689b1aad068e01751889f2f615c7db36d"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.29"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AutoHashEquals]]
git-tree-sha1 = "45bb6705d93be619b81451bb2006b7ee5d4e4453"
uuid = "15f4f7f2-30c1-5605-9d31-71845cf9641f"
version = "0.2.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "89e0654ed8c7aebad6d5ad235d6242c2d737a928"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.3"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "44dbf560808d49041989b8a96cae4cffbeb7966a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.11"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "b86ac2c5543660d238957dbde5ac04520ae977a7"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "02aa26a4cf76381be7f66e020a3eddeb27b0a092"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "d9a8f86737b665e15a9641ecbac64deef9ce6724"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.23.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "e460f044ca8b99be31d35fe54fc33a5c33dd8ed7"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.9.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "bf6570a34c850f99407b494757f5d7ad233a7257"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.5"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "8c86e48c0db1564a1d49548d3515ced5d604c408"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.9.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "f9d7112bfff8a19a3a4ea4e03a8e6a91fe8456bf"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.3"

[[deps.CoupledFields]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "6c9671364c68c1158ac2524ac881536195b7e7bc"
uuid = "7ad07ef1-bdf2-5661-9d2b-286fd4296dac"
version = "0.2.0"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataDeps]]
deps = ["BinaryProvider", "HTTP", "Libdl", "Reexport", "SHA", "p7zip_jll"]
git-tree-sha1 = "e299d8267135ef2f9c941a764006697082c1e7e8"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.8"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "b6def76ffad15143924a2199f72a5cd883a2e8a9"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.9"
weakdeps = ["SparseArrays"]

    [deps.Distances.extensions]
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "938fe2981db009f531b6332e31c58e9584a2f9bd"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.100"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Documenter]]
deps = ["Base64", "Dates", "DocStringExtensions", "InteractiveUtils", "JSON", "LibGit2", "Logging", "Markdown", "REPL", "Test", "Unicode"]
git-tree-sha1 = "395fa1554c69735802bba37d9e7d9586fd44326c"
uuid = "e30172f5-a6a5-5a46-863b-614d45cd2de4"
version = "0.24.11"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.Embeddings]]
deps = ["AutoHashEquals", "DataDeps", "GoogleDrive", "Statistics"]
git-tree-sha1 = "46da025753832eac7f10e6d761e29a7bb60bc40c"
uuid = "c5bfea45-b7f1-5224-a596-15500f5db411"
version = "0.4.2"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "b4fbdd20c889804969571cc589900803edda16b7"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "a20eaa3ad64254c61eeb5f230d9306e937405434"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.6.1"
weakdeps = ["SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "97829cfda0df99ddaeaafb5b370d6cab87b7013e"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.8.3"

[[deps.Gadfly]]
deps = ["Base64", "CategoricalArrays", "Colors", "Compose", "Contour", "CoupledFields", "DataAPI", "DataStructures", "Dates", "Distributions", "DocStringExtensions", "Hexagons", "IndirectArrays", "IterTools", "JSON", "Juno", "KernelDensity", "LinearAlgebra", "Loess", "Measures", "Printf", "REPL", "Random", "Requires", "Showoff", "Statistics"]
git-tree-sha1 = "d546e18920e28505e9856e1dfc36cff066907c71"
uuid = "c91e804a-d5a3-530f-b6f0-dfbca275c004"
version = "1.4.0"

[[deps.GoogleDrive]]
deps = ["DataDeps", "Dates", "Documenter", "HTTP", "Random"]
git-tree-sha1 = "6563e7c46ff2085937a5d9dd8eedac0c607a5a5e"
uuid = "91feb7a0-3508-11ea-1e8e-afea2c1c9a19"
version = "0.1.1"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "1cf1d7dcb4bc32d7b4a5add4232db3750c27ecb4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.8.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "MbedTLS", "Sockets"]
git-tree-sha1 = "c7ec02c4c6a039a98a15f955462cd7aea5df4508"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.8.19"

[[deps.Hexagons]]
deps = ["Test"]
git-tree-sha1 = "de4a6f9e7c4710ced6838ca906f81905f7385fd6"
uuid = "a1b4810d-1bce-5fbd-ac56-80944d57a21f"
version = "0.2.0"

[[deps.HistogramThresholding]]
deps = ["ImageBase", "LinearAlgebra", "MappedArrays"]
git-tree-sha1 = "7194dfbb2f8d945abdaf68fa9480a965d6661e69"
uuid = "2c695a8d-9458-5d45-9878-1b8a99cf7853"
version = "0.3.1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "eb8fed28f4994600e29beef49744639d985a04b2"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.16"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageBinarization]]
deps = ["HistogramThresholding", "ImageCore", "LinearAlgebra", "Polynomials", "Reexport", "Statistics"]
git-tree-sha1 = "f5356e7203c4a9954962e3757c08033f2efe578a"
uuid = "cbc4b850-ae4b-5111-9e64-df94c024a13d"
version = "0.3.0"

[[deps.ImageContrastAdjustment]]
deps = ["ImageBase", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "eb3d4365a10e3f3ecb3b115e9d12db131d28a386"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.12"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "fc5d1d3443a124fde6e92d0260cd9e064eba69f8"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.1"

[[deps.ImageCorners]]
deps = ["ImageCore", "ImageFiltering", "PrecompileTools", "StaticArrays", "StatsBase"]
git-tree-sha1 = "24c52de051293745a9bad7d73497708954562b79"
uuid = "89d5987c-236e-4e32-acd0-25bd6bd87b70"
version = "0.1.3"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "08b0e6354b21ef5dd5e49026028e41831401aca8"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.17"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "432ae2b430a18c58eb7eca9ef8d0f2db90bc749c"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.8"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "bca20b2f5d00c4fbc192c3212da8fa79f4688009"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.7"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "b0b765ff0b4c3ee20ce6740d843be8dfce48487c"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.3.0"

[[deps.ImageMagick_jll]]
deps = ["JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1c0a2295cca535fabaf2029062912591e9b61987"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.10-12+3"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "6f0a801136cb9c229aebea0df296cdcd471dbcd1"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.5"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "PrecompileTools", "Statistics"]
git-tree-sha1 = "783b70725ed326340adf225be4889906c96b8fd1"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.7"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "3ff0ca203501c3eedde3c6fa7fd76b703c336b5f"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.8.2"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "7ec124670cbce8f9f0267ba703396960337e54b5"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.10.0"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageBinarization", "ImageContrastAdjustment", "ImageCore", "ImageCorners", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "d438268ed7a665f8322572be0dabda83634d5f45"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.26.0"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "be8e690c3973443bec584db3346ddc904d4884eb"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ad37c091f7d7daf900963171600d7c1c5c3ede32"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalSets]]
deps = ["Dates", "Random"]
git-tree-sha1 = "8e59ea773deee525c99a8018409f64f19fb719e6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.7"
weakdeps = ["Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "aa6ffef1fd85657f4999030c52eaeec22a279738"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.33"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "327713faef2a3e5c80f96bf38d1fa26f7a6ae29e"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "88b8f66b604da079a627b6fb2860d3704a6729a1"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.14"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "9403bfea9bc9acc9c7d803a1b39d0a668ed40f03"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.6.2"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "ArrayInterfaceCore", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "c88a4afe1703d731b1c4fdf4e3c7e77e3b176ea2"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.165"

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.LoopVectorization.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "eb006abbd7041c28e0d16260e50a24f8f9104913"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "1130dbe1d5276cb656f6e1094ce97466ed700e5a"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "9b02b27ac477cad98114584ff964e3052f656a0f"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f6cf8e7944e50901594838951729a1861e668cb8"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.2"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase"]
git-tree-sha1 = "3aa2bb4982e575acd7583f01531f241af077b163"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.2.13"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "ee094908d720185ddbdc58dbe0c1cbe35453ec7a"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "ae36206463b2395804f2787ffe172f44452b538d"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.8.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "da095158bdc8eaccb7890f9884048555ab771019"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.4"

[[deps.RCall]]
deps = ["CategoricalArrays", "Conda", "DataFrames", "DataStructures", "Dates", "Libdl", "Missings", "REPL", "Random", "Requires", "StatsModels", "WinReg"]
git-tree-sha1 = "d9310ed05c2ff94c4e3a545a0e4c58ed36496179"
uuid = "6f49c342-dc21-5d91-9882-a32aef131414"
version = "0.13.16"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "54ccb4dbab4b1f69beb255a2c0ca5f65a9c82f08"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.5.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "4b8586aece42bee682399c4c4aee95446aa5cd19"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.39"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "4b33e0e081a825dbfaf314decf58fa47e53d6acb"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "f295e0a1da4ca425659c57441bcb59abb035a4bc"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.8"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "Requires", "SnoopPrecompile", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "33040351d2403b84afce74dae2e22d3f5b18edcb"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.4.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "9cabadf6e7cd2349b6cf49f1915ad2028d65e881"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.2"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "8cc7a5385ecaa420f0b3426f9b0135d0df0638ed"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.2"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "8621f5c499a8aa4aa970b1ae381aae0ef1576966"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.4"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "b182207d4af54ac64cbc71797765068fdeff475d"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.64"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WinReg]]
git-tree-sha1 = "cd910906b099402bcc50b3eafa9634244e5ec83b"
uuid = "1b915085-20d7-51cf-bf83-8f477d6f5128"
version = "1.0.0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═bc1220d8-3e5f-11ee-1966-0b1b5ba18a16
# ╠═85c8b5a1-bdca-4d55-beba-537add4ed507
# ╠═aa460f57-0bc6-444a-9118-66842c9f66bf
# ╟─cbe83826-1f31-41ee-a15c-2a725b7760a0
# ╠═8aa433bb-3775-4bc3-8ced-f8d3a78e37b4
# ╠═aac20471-7c34-40c2-8253-27cae3d9cfac
# ╠═703c9440-6d2b-411a-9e82-4c42cb8b6c43
# ╠═43a0c9a6-c898-40dc-9595-262fce56a8c4
# ╠═0add1b77-e3b9-405b-880c-c01e5eacc70a
# ╠═954a9af2-f800-450b-8d50-55ddea080d2d
# ╠═aa1a9fb4-c62f-4c04-8d50-90ee66636983
# ╠═25e59a93-d5cc-42cd-80c3-735f05b353f8
# ╠═f8b89ca5-2efe-4504-9185-698814f65354
# ╠═4d66c955-be09-4d42-8c74-7dbc82c17303
# ╟─dc5b7e52-26be-45f3-8f84-ea744e1d0cad
# ╠═b1177c92-dded-4d2d-9fb3-b5f799d4333f
# ╟─fa5477e8-80a8-4253-a167-e216d12d200c
# ╠═1ee0c8e1-f930-4350-83d9-39e6d304b7b1
# ╟─0baf238b-bbc0-4e5a-9089-ebedc05149b9
# ╠═90ce10d6-1715-49f7-8595-cfd54250868c
# ╟─d92a132f-c765-4885-8a1f-181e03ca5834
# ╟─faf447b5-75d3-418b-98da-3f4aa23fa97a
# ╠═0fc7b8ef-af3e-453e-9415-d4c755a9b45d
# ╠═bcb6af3b-d094-4066-9c4f-1f5d022d404a
# ╠═8017a28f-b0b7-4d77-b8c4-1a61c35a95ee
# ╠═aee253e3-7e48-41bd-a98d-aeb055ce94f6
# ╠═2dcf7754-8f58-4db5-a6cf-140ce2d5fe8d
# ╟─4ea7cb60-d418-4602-87ef-c007bf2469f5
# ╠═722142df-a242-483f-b8c2-b217b2a7895a
# ╠═5f8ef85b-4c6e-4997-9af8-8b72aa0fb79f
# ╠═a7a64ed0-8b66-4036-9bdd-2a99cd3802c0
# ╠═e7255ef1-bad0-4655-bcdc-3cc62329c574
# ╠═8ef0abcf-622d-472f-8665-ba6bd382f423
# ╠═cd562d4e-d23e-4721-860d-d3d5c761abb0
# ╟─04cdab51-28e7-4868-be7e-acefb3b300c2
# ╟─2f2e73ca-af62-4ac9-9a87-edd4f8dca689
# ╠═9dd11ecc-ea4e-45a9-91e5-e3e741e345b0
# ╠═34b2a1da-b788-4c05-8204-3358ead5814e
# ╠═a80ea0e9-d7a3-4ce8-b495-535dd03f4346
# ╠═4a2e3549-eac4-4137-8dab-dd947390e047
# ╠═f695a9a5-646c-4f17-868a-6f145415b831
# ╠═182732a1-f60a-47c5-a65f-3e55aa41c4c1
# ╠═336550e8-c077-4c74-b90a-d6d0da8a2ab6
# ╟─5d4ac17c-2d35-4ab7-b6ec-7f6aa36c1ec5
# ╟─3652a28d-fa65-47e1-82b2-84024477ecee
# ╠═74233fe9-8b20-4f79-9ec8-f49d9c8218e6
# ╟─c19c4718-a280-4fbe-add8-81ee41ee234b
# ╟─ce4eb397-0ea0-44c0-9447-f86837e7ee4c
# ╟─353c8c7d-7c95-4f3e-919d-9f31c0b39ab2
# ╠═9319a596-519f-4f3c-ac4e-f77335e4c708
# ╠═c11d45ee-469e-4343-8579-3ddfd6f022ac
# ╠═604a02fb-4a05-4d69-b250-0e9bfcd98a89
# ╠═133fde06-00d7-41f9-b4fc-7502dd8f22dd
# ╠═90e67763-9aa5-4f90-a5f0-38513b310cc6
# ╠═57db18a6-d35e-45ac-bc8b-b77699c474ce
# ╠═783bf0f0-0fd1-408b-8e8c-6ea7490bda16
# ╠═967985f9-6685-4b65-8d2c-521acc93cbed
# ╠═af7afb1c-51e8-41e6-8a6a-1d4da586864d
# ╟─14cf911b-6346-45da-9509-2f0de0e5b6ac
# ╠═a1b42e8f-331b-44a4-94c2-ea801a3209a0
# ╟─310c5e3e-f7b2-425f-886c-bb184dc159df
# ╟─0b9e4ba3-eb25-43de-b3e1-8f99f47a56a2
# ╟─baada914-8cd6-4499-9bbe-53ff3f813ffc
# ╟─835e1e12-2184-435e-a490-230d077efd16
# ╟─fab962cd-45cc-46cf-bd6b-a36a91813273
# ╟─6f188c70-f27c-4189-bb58-20d253e08c57
# ╟─e244fa4a-6f34-4e13-99c5-978942de18e2
# ╠═d817ae43-fdfd-40c5-85e8-900d300c889e
# ╠═d8527fe6-3ea6-4f51-95d5-f663cf5aafc9
# ╠═3b8f0266-4900-4b68-b316-9cbf303cb5ae
# ╠═d8eba36c-263a-47f4-97de-083900ac45e6
# ╠═b375285d-44ff-409a-99cb-79214770a70c
# ╟─ea0d2ff6-576f-4eae-aec1-39d07dc9a97a
# ╟─4a6db4ad-9dc4-4b75-a700-f6855da6f352
# ╟─fe30cb66-2c23-4c1d-bdf4-62945b1bc4e8
# ╠═4d71aac6-2c62-439b-abe1-943b81d3d78a
# ╠═4c90e1ac-9c11-4278-b800-2f29b85841a2
# ╠═79e14648-19a0-43a6-beba-a5e1de818cce
# ╠═ace7343d-842a-4f59-a5bd-33cd62079b9c
# ╟─f8f299c4-39d1-489a-800a-d67529753395
# ╟─80b3b3d3-02f4-4d79-aaea-fcef3c1ad8d9
# ╟─45472a6a-c7d7-4a2e-87f0-4659e85ecd0c
# ╠═1baf1101-aaaf-453c-aa1f-1c3cb932e1a5
# ╠═fdea0e39-2314-4b80-998d-47a756ecac86
# ╠═0e2f3868-c1f7-4a4d-b16a-c794fffc4060
# ╠═59babe1a-a922-4ddb-858e-ad25153997b0
# ╟─d88d90f6-f054-4ba7-8a27-6d2a012fcabb
# ╠═32b991cc-042d-4e69-8884-1157a0b64d03
# ╠═6a690490-d1a8-4795-89c7-01759b9ff3f3
# ╠═ffc0df4d-2ba3-4071-9bc0-6a698f400413
# ╠═8fb3a9fa-e6f0-44e1-8191-579fa1b2999d
# ╟─26d79d37-f6bc-454e-af54-039f40f23e25
# ╟─f17c0070-5413-4e18-b22b-fd8187cda2a4
# ╠═f319b01c-3c4a-4414-8724-c2dc60819a41
# ╠═86cb9d38-6498-45db-9e44-11f5872f7b7e
# ╠═5ffe2933-554a-43f9-9c28-fe7e1aa2bb78
# ╟─4a127d95-a24a-4f2e-b362-d01342140558
# ╟─a4fbf469-bb7a-41d4-92ff-9d406ab206b8
# ╟─2b2e24b9-d583-42e9-9165-ec5bf891fe40
# ╠═b0547f88-b11a-4775-96c7-00903ac3dbed
# ╠═4513c56e-c184-4bf1-bdbc-74ada2e89900
# ╟─255ff2b1-8cda-48d2-b2d6-a29009131f27
# ╠═485ebe6b-023c-423a-8be7-8fe6c964a416
# ╠═0efb986a-3701-48d1-9823-a1522a2c66c0
# ╠═6ef3626b-37e9-4602-9cb2-a23067f8a73f
# ╟─975da245-9ce2-4397-85e5-80b1d4a4c1ce
# ╠═0264d362-b1bc-46ae-92fd-914359d27e3d
# ╠═c408f9cc-a21f-40a0-94e9-23f7c9dc7673
# ╠═abfd806b-bf99-4cd3-a33c-fc6690265c70
# ╟─67265e5a-42da-479c-ab05-3148f3423243
# ╟─07b77a36-ddb9-47cb-998a-f3e6e2c2a954
# ╠═336ec162-a6ad-49b7-8b97-1be5487a3b13
# ╠═d6e7ee63-1be4-482f-b5c0-9c558f37bf22
# ╠═d90ab635-3c95-44ab-ac4c-902a1585bad6
# ╟─4ba3eb73-5c64-458d-9b7d-8d1ee8c844bb
# ╟─47b34b3d-7d46-482b-9b0c-07a09f07be5c
# ╠═426ed389-7df8-4b29-b8e8-b418b2815cc4
# ╠═0dea6480-5116-4458-bafd-b4ea1afa6214
# ╠═b0a2e043-7b40-4175-8f28-0a08928edb02
# ╠═3b33cf52-37ae-4cbc-83d3-ecc09f9f9e2b
# ╠═399b874d-bee4-40af-a66e-8f26de2b859b
# ╟─79668bfc-e5d5-4bda-b54d-c0b34749ca6a
# ╠═5224ef88-c0c6-4db3-86b9-90a610a7aa79
# ╠═dd8aecbd-c0d2-4ef7-aadf-09e32663303d
# ╠═5a5eb6a5-374d-47d8-88f3-a895404faeb7
# ╠═56090994-128a-4476-b506-7920a938180a
# ╠═eedd241b-9ad9-495f-b75d-74ed9c05528f
# ╟─6b695d46-3eb5-4961-8181-af517341db5e
# ╠═185a8efd-075f-4991-b3e8-b2c25cb6e830
# ╠═519669dc-994c-4779-985c-f4d8e67aeb09
# ╠═29cbb72a-abf6-43f2-a4d8-0a242c460058
# ╟─67a448ba-c544-4b66-9614-9fe0ef0eda80
# ╠═d1747ce8-4865-414a-917c-68b57120fcce
# ╠═3648bbd7-6b08-435e-9cac-693338d8486e
# ╠═fab8dd11-bb25-4d3b-8d5b-d3ef78e2c4cc
# ╠═f2c46501-cfa2-4445-b15b-f72fdd419aa4
# ╠═53d51396-5b22-4d79-a37b-0b91021a1b6d
# ╠═6b989965-bd57-4d11-b26e-486d71966fc8
# ╠═21e28b5a-09da-47db-95e2-3dfe5b28d281
# ╠═570b0213-0c7e-4f42-91c0-5ca00dfeb2fd
# ╠═a85440cf-110b-4459-adcd-a45661aaa17a
# ╠═09521601-3be8-403c-b78c-e9697e30d775
# ╠═8314fd26-c7e9-46ce-8d43-13ea6572487b
# ╠═d558a989-b2b6-46a8-8d3d-c13dfa6f1cb1
# ╠═6096a53f-f5b4-42d7-9c22-9455f18f86f2
# ╠═515738e6-868c-43dd-a063-c0084ee7198f
# ╠═e98cdc7f-8e81-4298-87e1-c7769b0c3878
# ╠═9b6b84e1-f9eb-49cf-8d7f-248af694226b
# ╠═d9259412-4ccb-447f-a34b-06dc20659baf
# ╠═013d1401-171a-4219-af04-c5e7ef002e5c
# ╠═e2deebe4-fee7-49dc-96d2-558c24af5c40
# ╠═d0d411be-75f2-4126-8f64-2c93b43fe365
# ╠═6526864b-fe7b-40cf-b81e-e156e09830f7
# ╠═7105694b-2121-445d-a4a8-79e74f01ee96
# ╠═67ec6b84-9f87-4ab0-8994-fa714e1a23e8
# ╠═d1b0f02a-5924-4bac-8a84-29b167aed3e5
# ╠═80efe2f1-2df5-42c9-8be5-f9a373037bbd
# ╠═e8d97c67-2cfe-4ee3-a65a-355deac7f46d
# ╠═fa5a8d29-531b-40d0-8953-9acbd7774754
# ╠═14b7ef6d-7a76-4e14-94e0-1a9b0ac9e813
# ╠═4d14f654-1d51-48e7-86a8-e046ee935daf
# ╠═a25a9b13-a7c8-4f1e-a79b-c78e2b2e2e6e
# ╠═24276b2f-d710-4798-a16d-f1bbbd418c81
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
