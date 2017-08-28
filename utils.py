def theme_cognoma(fontsize_mult=1):   
    import plotnine as gg
    
    return (gg.theme_bw(base_size = 14 * fontsize_mult) +
        gg.theme(
          line = gg.element_line(color = "#4d4d4d"), 
          rect = gg.element_rect(fill = "white", color = None), 
          text = gg.element_text(color = "black"), 
          axis_ticks = gg.element_line(color = "#4d4d4d"),
          legend_key = gg.element_rect(color = None), 
          panel_border = gg.element_rect(color = "#4d4d4d"),  
          panel_grid = gg.element_line(color = "#b3b3b3"), 
          panel_grid_major_x = gg.element_blank(),
          panel_grid_minor = gg.element_blank(),
          strip_background = gg.element_rect(fill = "#FEF2E2", color = "#4d4d4d"),
          axis_text = gg.element_text(size = 12 * fontsize_mult, color="#4d4d4d"),
          axis_title_x = gg.element_text(size = 13 * fontsize_mult, color="#4d4d4d"),
          axis_title_y = gg.element_text(size = 13 * fontsize_mult, color="#4d4d4d")
    ))

def get_model_coefficients(classifier, feature_set, covariate_names):
    """
    Extract the feature names and associate them with the coefficient values
    in the final classifier object.
    * Only works for expressions only model with PCA, covariates only model,
        and a combined model
    * Assumes the PCA features come before any covariates that are included
    * Sorts the final dataframe by the absolute value of the coefficients
    
    Args:
        classifier: the final sklearn classifier object 
        feature_set: string of the model's name {expressions, covariates, full}
        covariate_names: list of the names of the covariate features matrix
    
    Returns:
        pandas.DataFrame: mapping of feature name to coefficient value
    """
    import pandas as pd
    import numpy as np
    
    coefs = classifier.coef_[0]   
    
    if feature_set=='expressions':
        features = ['PCA_%d' %cf for cf in range(len(coefs))]
    elif feature_set=='covariates': 
        features = covariate_names
    else:        
        features = ['PCA_%d' %cf for cf in range(len(coefs) - len(covariate_names))]
        features.extend(covariate_names)
     
    coef_df = pd.DataFrame({'feature': features, 'weight': coefs})  
        
    coef_df['abs'] = coef_df['weight'].abs()
    coef_df = coef_df.sort_values('abs', ascending=False)
    coef_df['feature_set'] = feature_set
    
    
    return coef_df

def get_genes_coefficients(pca_object, classifier_object,
                           expression_df, expression_genes_df,
                           num_covariates=None):
    """Identify gene coefficients from classifier after pca.

    Args:
        pca_object: The pca object from running pca on the expression_df.
        classifier_object: The logistic regression classifier object.
        expression_df: The original (pre-pca) expression data frame.
        expression_genes_df: The "expression_genes" dataframe used for gene
                             names.
        num_covariates: Optional, only needed if PCA was only performed on a
                        subset of the features. This should be the number of
                        features that PCA was not performed on. This function
                        assumes that the covariates features were at the end.

    Returns:
        gene_coefficients_df: A dataframe with entreze gene-ID, gene name,
                            coefficient abbsolute value of coefficient, and 
                            gene description. The dataframe is sorted by 
                            absolute value of coefficient.
    """

    import pandas as pd

    # Get the classifier coefficients.
    if num_covariates:
        coefficients = classifier_object.coef_[0][0:-num_covariates]
    else:
        coefficients = classifier_object.coef_[0]
    # Get the pca weights.
    weights = pca_object.components_
    # Combine the coefficients and weights.
    gene_coefficients = weights.T @ coefficients.T
    # Create the dataframe with correct index
    gene_coefficients_df = pd.DataFrame(gene_coefficients, columns=['weight'])
    gene_coefficients_df.index = expression_df.columns
    gene_coefficients_df.index.name = 'entrez_id'
    expression_genes_df.index = expression_genes_df.index.map(str)
    # Add gene symbol and description
    gene_coefficients_df['symbol'] = expression_genes_df['symbol']
    gene_coefficients_df['description'] = expression_genes_df['description'] 
    # Add absolute value and sort by highest absolute value.
    gene_coefficients_df['abs'] = gene_coefficients_df['weight'].abs()
    gene_coefficients_df.sort_values(by='abs', ascending=False, inplace=True)
    # Reorder columns
    gene_coefficients_df = gene_coefficients_df[['symbol', 'weight', 'abs',
                                                'description']]
    return(gene_coefficients_df)
