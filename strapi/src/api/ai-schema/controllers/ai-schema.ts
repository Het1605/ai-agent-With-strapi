const mapFieldToStrapiAttribute = (field: any) => {
    // 1. Determine base type (Handle Strapi v5 number granularity)
    let type = field.type;
    if (type === 'number') {
        type = field.numberSize || 'integer';
    }

    // 2. Initialize base config
    const config: any = { type };

    // 3. Map Global Options (Present in most types)
    const globalOptions = ['required', 'unique', 'default', 'private', 'configurable'];
    globalOptions.forEach(opt => {
        if (field[opt] !== undefined) config[opt] = field[opt];
    });

    // 4. Map Type-Specific Validations/Options
    switch (type) {
        case 'string':
        case 'text':
        case 'richtext':
        case 'email':
        case 'password':
        case 'uid':
            if (field.minLength !== undefined) config.minLength = field.minLength;
            if (field.maxLength !== undefined) config.maxLength = field.maxLength;
            if (field.regex !== undefined) config.regex = field.regex;
            if (type === 'uid' && field.targetField) config.targetField = field.targetField;
            break;

        case 'integer':
        case 'biginteger':
        case 'decimal':
        case 'float':
            if (field.min !== undefined) config.min = field.min;
            if (field.max !== undefined) config.max = field.max;
            break;

        case 'enumeration':
            config.enum = field.enum || [];
            break;

        case 'media':
            config.multiple = !!field.multiple;
            config.allowedTypes = field.allowedTypes || ['images', 'files', 'videos', 'audios'];
            break;

        case 'relation':
            config.relation = field.relation;
            config.target = field.target;
            if (field.targetAttribute) config.targetAttribute = field.targetAttribute;
            break;

        default:
            // Other types (blocks, json, boolean, date) usually only use global options
            break;
    }

    return config;
};

export default {
    async createCollection(ctx: any) {
        const { collectionName, singularName, pluralName, displayName, fields } = ctx.request.body;

        if (!collectionName || !singularName || !pluralName) {
            return ctx.badRequest('collectionName, singularName, and pluralName are required');
        }

        // Use the exact names provided by the Python Lean Agent
        const actualSingular = singularName.toLowerCase();
        const actualPlural = pluralName.toLowerCase();
        const actualDisplayName = displayName || actualSingular;

        // @ts-ignore
        strapi.log.info(`[StrapiBridge] Creating collection: singular=${actualSingular}, plural=${actualPlural}, display=${actualDisplayName}`);
        
        // UID format: api::singular.singular
        const uid = `api::${actualSingular}.${actualSingular}`;

        // Check if it already exists to avoid 500 errors
        // @ts-ignore
        if (strapi.contentTypes[uid]) {
            return ctx.badRequest(`Collection "${uid}" already exists.`);
        }

        const attributes: any = {};
        for (const field of fields) {
            attributes[field.name] = mapFieldToStrapiAttribute(field);
        }

        try {
            // @ts-ignore
            const contentTypeService = strapi.plugin('content-type-builder').service('content-types');

            // Strapi v5 createContentType expects an object with contentType and components
            const result = await contentTypeService.createContentType({
                contentType: {
                    displayName: actualDisplayName,
                    singularName: actualSingular,
                    pluralName: actualPlural,
                    kind: 'collectionType',
                    attributes,
                },
                components: [],
            });

            // The service call triggers a hot-reload in development mode.
            // We respond immediately to ensure the client gets a success message
            // before the server process might restart.
            ctx.send({
                message: 'Collection created successfully',
                uid: result.uid,
            });
        } catch (error: any) {
            // @ts-ignore
            strapi.log.error(error);

            if (error.message === 'contentType.alreadyExists') {
                return ctx.badRequest('Collection already exists');
            }

            ctx.internalServerError(`Failed to create collection: ${error.message}`);
        }
    },

    async modifySchema(ctx: any) {
        const { operation, collection } = ctx.request.body;
        const data = ctx.request.body.data || {};

        // ── Validate common required fields ───────────────────────────
        if (!operation) {
            return ctx.badRequest('"operation" is required. Supported: add_column, update_collection, update_column, delete_column');
        }
        if (!collection) {
            return ctx.badRequest('"collection" is required');
        }

        // Use the exact singular name provided by the Python side (Lean Agent)
        const singularName = collection.toLowerCase();
        const uid = `api::${singularName}.${singularName}`;

        // @ts-ignore
        strapi.log.info(`[StrapiBridge] Modifying collection: ${uid} (operation: ${operation})`);

        // @ts-ignore
        const existingContentType = strapi.contentTypes[uid];
        if (!existingContentType) {
            return ctx.badRequest(`Collection "${uid}" does not exist. Create it first.`);
        }

        // @ts-ignore
        const contentTypeService = strapi.plugin('content-type-builder').service('content-types');

        // Helper to call editContentType and send a uniform success response
        const applyEdit = async (updatedAttributes: any, extraInfo: any = {}) => {
            const info = existingContentType.info || {};
            await contentTypeService.editContentType(uid, {
                contentType: {
                    displayName: info.displayName || singularName,
                    singularName: info.singularName || singularName,
                    pluralName: info.pluralName || singularName,
                    kind: 'collectionType',
                    attributes: updatedAttributes,
                },
                components: [],
            });
            ctx.send({ message: 'Schema updated successfully', ...extraInfo });
        };

        try {
            const existingAttributes: any = { ...(existingContentType.attributes || {}) };

            switch (operation) {

                // ── 1. ADD COLUMN ─────────────────────────────────────
                case 'add_column': {
                    const fields: any[] = data.fields || [];
                    if (!Array.isArray(fields) || fields.length === 0) {
                        return ctx.badRequest('"data.fields" must be a non-empty array for add_column');
                    }

                    const added: string[] = [];
                    const skipped: string[] = [];
                    const invalid: string[] = [];

                    for (const field of fields) {
                        if (!field.name || !field.type) {
                            invalid.push(field.name || '(unnamed)');
                            continue;
                        }
                        if (existingAttributes[field.name]) {
                            skipped.push(field.name);
                            continue;
                        }
                        existingAttributes[field.name] = mapFieldToStrapiAttribute(field);
                        added.push(field.name);
                    }

                    if (invalid.length > 0) {
                        return ctx.badRequest(`Fields missing name or type: ${invalid.join(', ')}`);
                    }
                    if (added.length === 0) {
                        return ctx.badRequest(
                            skipped.length > 0
                                ? `All provided fields already exist: ${skipped.join(', ')}`
                                : 'No valid new fields to add'
                        );
                    }

                    const extra: any = { collection: uid, added };
                    if (skipped.length > 0) extra.skipped = skipped;
                    await applyEdit(existingAttributes, extra);
                    break;
                }

                // ── 2. UPDATE COLLECTION ──────────────────────────────
                // Only updates display-level info (displayName, description).
                // Does NOT rename the UID, folder, or singularName.
                case 'update_collection': {
                    // ── Branch: DELETE entire collection ─────────────
                    if (data.delete === true) {
                        await contentTypeService.deleteContentType(uid);
                        ctx.send({
                            message: 'Collection deleted successfully',
                            collection: uid,
                            deleted: true,
                        });
                        break;
                    }

                    // ── Branch: UPDATE collection settings ────────────
                    if (!data.displayName && !data.description) {
                        return ctx.badRequest('"data.displayName" or "data.description" is required for update_collection (or pass "delete": true to delete the collection)');
                    }

                    const info = existingContentType.info || {};
                    // @ts-ignore
                    await contentTypeService.editContentType(uid, {
                        contentType: {
                            displayName: data.displayName || info.displayName || singularName,
                            singularName: info.singularName || singularName,
                            pluralName: data.pluralName || info.pluralName || singularName,
                            description: data.description || info.description || '',
                            kind: 'collectionType',
                            attributes: existingAttributes,
                        },
                        components: [],
                    });
                    ctx.send({
                        message: 'Schema updated successfully',
                        collection: uid,
                        updated: Object.fromEntries(
                            ['displayName', 'description']
                                .filter(k => data[k] !== undefined)
                                .map(k => [k, data[k]])
                        ),
                    });
                    break;
                }


                // ── 3. UPDATE COLUMN ───────────────────────────────────
                case 'update_column': {
                    const fieldName = data.field;
                    const updates = data.updates;

                    if (!fieldName) {
                        return ctx.badRequest('"data.field" is required for update_column');
                    }
                    if (!updates || typeof updates !== 'object') {
                        return ctx.badRequest('"data.updates" must be an object for update_column');
                    }
                    if (!existingAttributes[fieldName]) {
                        return ctx.badRequest(`Field "${fieldName}" does not exist in collection "${uid}"`);
                    }

                    // Merge updates into the existing field config; type cannot be changed.
                    existingAttributes[fieldName] = {
                        ...existingAttributes[fieldName],
                        ...updates,
                        type: existingAttributes[fieldName].type,  // type is immutable
                    };

                    await applyEdit(existingAttributes, {
                        collection: uid,
                        field: fieldName,
                        applied: updates,
                    });
                    break;
                }

                // ── 4. DELETE COLUMN ───────────────────────────────────
                case 'delete_column': {
                    const fieldName = data.field;
                    if (!fieldName) {
                        return ctx.badRequest('"data.field" is required for delete_column');
                    }
                    if (!existingAttributes[fieldName]) {
                        return ctx.badRequest(`Field "${fieldName}" does not exist in collection "${uid}"`);
                    }

                    delete existingAttributes[fieldName];

                    await applyEdit(existingAttributes, {
                        collection: uid,
                        deleted: fieldName,
                    });
                    break;
                }

                default:
                    return ctx.badRequest(
                        `Unsupported operation "${operation}". Allowed: add_column, update_collection, update_column, delete_column`
                    );
            }

        } catch (error: any) {
            // @ts-ignore
            strapi.log.error('modifySchema error:', error);
            ctx.internalServerError(`Failed to modify schema: ${error.message}`);
        }
    },

    async fieldRegistry(ctx: any) {
        try {
            const fieldMap: Record<string, Set<string>> = {};

            // @ts-ignore
            const contentTypes = strapi.contentTypes;

            for (const uid in contentTypes) {
                const contentType = contentTypes[uid];
                const attributes = contentType.attributes;

                for (const attrName in attributes) {
                    const attr = attributes[attrName];
                    const type = attr.type;

                    if (!type) continue;

                    if (!fieldMap[type]) {
                        fieldMap[type] = new Set<string>();
                    }

                    // Dynamically collect all keys that are not 'type'
                    Object.keys(attr).forEach(key => {
                        if (key !== 'type') {
                            fieldMap[type].add(key);
                        }
                    });
                }
            }

            // Convert Sets to Arrays for JSON response
            const responseFields: Record<string, string[]> = {};
            for (const type in fieldMap) {
                responseFields[type] = Array.from(fieldMap[type]);
            }

            // Get list of all collection UIDs and names
            const collection_uids = Object.keys(contentTypes).filter(uid => uid.startsWith('api::'));
            const collections = collection_uids.map(uid => uid.split('.')[1]); // Extract 'employee' from 'api::employee.employee'

            return ctx.send({
                fields: responseFields,
                collections,
                collection_uids
            });
        } catch (error: any) {
            // @ts-ignore
            strapi.log.error('Field Registry Error:', error);
            return ctx.internalServerError(`Failed to retrieve field registry: ${error.message}`);
        }
    },

    async listContentTypes(ctx: any) {
        try {
            // @ts-ignore
            const contentTypes = strapi.contentTypes;
            const collections = [];

            for (const uid in contentTypes) {
                if (uid.startsWith('api::')) {
                    const ct = contentTypes[uid];
                    collections.push({
                        uid: uid,
                        name: ct.info.singularName,
                        displayName: ct.info.displayName
                    });
                }
            }

            return ctx.send({ collections });
        } catch (error: any) {
            // @ts-ignore
            strapi.log.error('listContentTypes error:', error);
            return ctx.internalServerError(`Failed to list content types: ${error.message}`);
        }
    },

    async getContentTypeSchema(ctx: any) {
        try {
            const { name } = ctx.params;
            const actualName = name.toLowerCase();
            const uid = `api::${actualName}.${actualName}`;

            // @ts-ignore
            const ct = strapi.contentTypes[uid];
            if (!ct) {
                return ctx.notFound(`Content type "${uid}" not found`);
            }

            return ctx.send({
                name: ct.info.singularName,
                displayName: ct.info.displayName,
                attributes: ct.attributes
            });
        } catch (error: any) {
            // @ts-ignore
            strapi.log.error('getContentTypeSchema error:', error);
            return ctx.internalServerError(`Failed to get content type schema: ${error.message}`);
        }
    },
};
